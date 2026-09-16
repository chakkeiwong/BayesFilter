"""Signed-log tail evaluation for the frozen Lane-B T2 untouched stream.

The ordinary route remains the numerical authority whenever the author RK4
program is representable in float64. Signed-log arithmetic is used only to
certify that a nonrepresentable row has FP64 extended-real zero likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.models import _zhao_cui_sir_austria_transition_mean_xla
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import tensor_sha256
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_tf import (
    LaneBT2LogNormalizerEstimate,
    t2_source_closure,
)


DTYPE = tf.float64
SCHEMA = "bayesfilter.zhao_cui_austria_sir_lane_b_t2_untouched_tail.v1"
TAIL_EVALUATOR_ID = "author_sir_step_signed_log_fp64_zero_certificate_v1"
ROOT = Path(__file__).resolve().parents[2]
AUTHOR_SIR_STEP = Path(
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/sir_austria/sir_step.mlx"
)
TAIL_REPAIR_PLAN = Path(
    "docs/plans/"
    "bayesfilter-zhao-cui-austria-sir-lane-b-t2-untouched-tail-repair-plan-2026-07-31.md"
)
FP64_STANDARDIZED_RESIDUAL_LOG_THRESHOLD = (
    0.5 * math.log(sys.float_info.max) + 20.0
)
_NEIGHBORS = (
    (1,),
    (0, 2, 3),
    (1, 3, 4, 5),
    (1, 2, 4),
    (2, 3, 5, 6, 8),
    (2, 4, 6),
    (4, 5, 7, 8),
    (6,),
    (4, 6),
)


def _negative_infinity_like(value: tf.Tensor) -> tf.Tensor:
    return tf.fill(tf.shape(value), tf.constant(float("-inf"), DTYPE))


def signed_log_from_real(value: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    values = tf.convert_to_tensor(value, DTYPE)
    sign = tf.sign(values)
    log_abs = tf.where(
        sign == 0.0, _negative_infinity_like(values), tf.math.log(tf.abs(values))
    )
    return sign, log_abs


def signed_log_to_real(value: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    sign, log_abs = value
    return sign * tf.exp(log_abs)


def signed_log_add(
    left: tuple[tf.Tensor, tf.Tensor],
    right: tuple[tf.Tensor, tf.Tensor],
) -> tuple[tf.Tensor, tf.Tensor]:
    left_sign, left_log = left
    right_sign, right_log = right
    output_shape = tf.broadcast_dynamic_shape(tf.shape(left_sign), tf.shape(right_sign))
    left_sign = tf.broadcast_to(left_sign, output_shape)
    left_log = tf.broadcast_to(left_log, output_shape)
    right_sign = tf.broadcast_to(right_sign, output_shape)
    right_log = tf.broadcast_to(right_log, output_shape)
    left_zero = left_sign == 0.0
    right_zero = right_sign == 0.0
    same_sign = left_sign == right_sign
    same_log = tf.reduce_logsumexp(tf.stack([left_log, right_log]), axis=0)
    left_larger = left_log > right_log
    larger_log = tf.maximum(left_log, right_log)
    smaller_log = tf.minimum(left_log, right_log)
    different_log = larger_log + tf.math.log1p(
        -tf.exp(smaller_log - larger_log)
    )
    equal_magnitude = left_log == right_log
    different_sign = tf.where(left_larger, left_sign, right_sign)
    result_sign = tf.where(
        left_zero,
        right_sign,
        tf.where(
            right_zero,
            left_sign,
            tf.where(
                same_sign,
                left_sign,
                tf.where(equal_magnitude, tf.zeros_like(left_sign), different_sign),
            ),
        ),
    )
    result_log = tf.where(
        left_zero,
        right_log,
        tf.where(
            right_zero,
            left_log,
            tf.where(
                same_sign,
                same_log,
                tf.where(equal_magnitude, _negative_infinity_like(left_log), different_log),
            ),
        ),
    )
    return result_sign, result_log


def signed_log_negate(
    value: tuple[tf.Tensor, tf.Tensor]
) -> tuple[tf.Tensor, tf.Tensor]:
    return -value[0], value[1]


def signed_log_multiply(
    left: tuple[tf.Tensor, tf.Tensor],
    right: tuple[tf.Tensor, tf.Tensor],
) -> tuple[tf.Tensor, tf.Tensor]:
    sign = left[0] * right[0]
    log_abs = tf.where(
        sign == 0.0,
        _negative_infinity_like(sign),
        left[1] + right[1],
    )
    return sign, log_abs


def signed_log_scale(
    value: tuple[tf.Tensor, tf.Tensor], scalar: float
) -> tuple[tf.Tensor, tf.Tensor]:
    factor = signed_log_from_real(tf.constant(float(scalar), DTYPE))
    return signed_log_multiply(value, factor)


def _signed_log_sum(
    values: tuple[tuple[tf.Tensor, tf.Tensor], ...]
) -> tuple[tf.Tensor, tf.Tensor]:
    if not values:
        raise ValueError("signed-log sum requires at least one term")
    result = values[0]
    for value in values[1:]:
        result = signed_log_add(result, value)
    return result


def _gather_component(
    value: tuple[tf.Tensor, tf.Tensor], index: int
) -> tuple[tf.Tensor, tf.Tensor]:
    return value[0][:, index], value[1][:, index]


def _signed_log_rhs(
    state: tuple[tf.Tensor, tf.Tensor]
) -> tuple[tf.Tensor, tf.Tensor]:
    susceptible = state[0][:, 0::2], state[1][:, 0::2]
    infectious = state[0][:, 1::2], state[1][:, 1::2]
    ds = []
    di = []
    for index, neighbors in enumerate(_NEIGHBORS):
        susceptible_neighbor = _signed_log_sum(
            tuple(_gather_component(susceptible, neighbor) for neighbor in neighbors)
            + (
                signed_log_scale(
                    signed_log_negate(_gather_component(susceptible, index)),
                    float(len(neighbors)),
                ),
            )
        )
        infectious_neighbor = _signed_log_sum(
            tuple(_gather_component(infectious, neighbor) for neighbor in neighbors)
            + (
                signed_log_scale(
                    signed_log_negate(_gather_component(infectious, index)),
                    float(len(neighbors)),
                ),
            )
        )
        infection = signed_log_scale(
            signed_log_multiply(
                _gather_component(susceptible, index),
                _gather_component(infectious, index),
            ),
            0.1,
        )
        ds.append(
            signed_log_add(
                signed_log_negate(infection),
                signed_log_scale(susceptible_neighbor, 0.5),
            )
        )
        di.append(
            _signed_log_sum(
                (
                    infection,
                    signed_log_scale(
                        signed_log_negate(_gather_component(infectious, index)),
                        18.0,
                    ),
                    signed_log_scale(infectious_neighbor, 0.5),
                )
            )
        )
    signs = []
    logs = []
    for susceptible_value, infectious_value in zip(ds, di):
        signs.extend([susceptible_value[0], infectious_value[0]])
        logs.extend([susceptible_value[1], infectious_value[1]])
    return tf.stack(signs, axis=1), tf.stack(logs, axis=1)


def signed_log_author_transition_mean(
    physical_previous: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Evaluate exactly the author four-stage/four-substep RK polynomial."""

    state = signed_log_from_real(tf.convert_to_tensor(physical_previous, DTYPE))
    for _ in range(4):
        k1 = _signed_log_rhs(state)
        k2 = _signed_log_rhs(signed_log_add(state, signed_log_scale(k1, 0.0025)))
        k3 = _signed_log_rhs(signed_log_add(state, signed_log_scale(k2, 0.0025)))
        # The half-step fourth stage is the author-source convention.
        k4 = _signed_log_rhs(signed_log_add(state, signed_log_scale(k3, 0.0025)))
        weighted = _signed_log_sum(
            (k1, signed_log_scale(k2, 2.0), signed_log_scale(k3, 2.0), k4)
        )
        state = signed_log_add(state, signed_log_scale(weighted, 0.005 / 6.0))
    return state


@dataclass(frozen=True)
class TailChunkResult:
    finite_z2: tf.Tensor
    log_likelihood: tf.Tensor
    transition_log_density: tf.Tensor
    nonrepresentable_mask: tf.Tensor
    zero_target_mask: tf.Tensor
    overflow_log_margin: tf.Tensor
    ordinary_relative_residual: tf.Tensor


def evaluate_t2_tail_chunk(
    *,
    z1: tf.Tensor,
    transition_noise: tf.Tensor,
    observation: tf.Tensor,
) -> TailChunkResult:
    """Evaluate ordinary rows directly and certify nonrepresentable tail rows."""

    previous = tf.convert_to_tensor(z1, DTYPE)
    noise = tf.convert_to_tensor(transition_noise, DTYPE)
    observed = tf.reshape(tf.convert_to_tensor(observation, DTYPE), [9])
    susceptible = tf.maximum(previous[:, 0::2], tf.constant(0.0, DTYPE))
    physical_previous = tf.reshape(
        tf.stack([susceptible, previous[:, 1::2]], axis=2), [-1, 18]
    )
    direct_mean = _zhao_cui_sir_austria_transition_mean_xla(
        tf.zeros([3], DTYPE), physical_previous
    )
    direct_z2 = direct_mean + noise
    representable = tf.reduce_all(tf.math.is_finite(direct_z2), axis=1)
    nonrepresentable = ~representable

    signed_mean = signed_log_author_transition_mean(physical_previous)
    signed_z2 = signed_log_add(signed_mean, signed_log_from_real(noise))
    signed_infectious = signed_z2[0][:, 1::2], signed_z2[1][:, 1::2]
    signed_residual = signed_log_add(
        signed_infectious,
        signed_log_negate(signed_log_from_real(observed[tf.newaxis, :])),
    )
    maximum_standardized_log_residual = tf.reduce_max(
        signed_residual[1] - tf.constant(math.log(10.0), DTYPE), axis=1
    )
    overflow_margin = (
        maximum_standardized_log_residual
        - tf.constant(FP64_STANDARDIZED_RESIDUAL_LOG_THRESHOLD, DTYPE)
    )
    if bool(tf.reduce_any(nonrepresentable & (overflow_margin <= 0.0)).numpy()):
        raise ValueError("nonrepresentable T2 row lacks FP64 zero-density certificate")

    reconstructed = signed_log_to_real(signed_mean)
    safe_direct = tf.where(representable[:, tf.newaxis], direct_mean, tf.zeros_like(direct_mean))
    safe_reconstructed = tf.where(
        representable[:, tf.newaxis], reconstructed, tf.zeros_like(reconstructed)
    )
    relative = tf.reduce_max(
        tf.abs(safe_reconstructed - safe_direct) / (1.0 + tf.abs(safe_direct)),
        axis=1,
    )
    if bool(tf.reduce_any(representable & (relative > 2e-11)).numpy()):
        raise ValueError("signed-log author transition disagrees on ordinary row")

    finite_placeholder = tf.where(
        representable[:, tf.newaxis], direct_z2, tf.zeros_like(direct_z2)
    )
    residual = observed[tf.newaxis, :] - finite_placeholder[:, 1::2]
    ordinary_likelihood = -0.5 * (
        tf.constant(9.0 * math.log(2.0 * math.pi * 100.0), DTYPE)
        + tf.reduce_sum(tf.square(residual), axis=1) / tf.constant(100.0, DTYPE)
    )
    log_likelihood = tf.where(
        representable,
        ordinary_likelihood,
        tf.fill(tf.shape(ordinary_likelihood), tf.constant(float("-inf"), DTYPE)),
    )
    invalid_likelihood = tf.math.is_nan(log_likelihood) | (
        tf.math.is_inf(log_likelihood) & (log_likelihood > 0.0)
    )
    if bool(tf.reduce_any(invalid_likelihood).numpy()):
        raise ValueError("tail-aware T2 likelihood contains NaN or positive infinity")
    transition_log_density = -0.5 * (
        tf.constant(18.0 * math.log(2.0 * math.pi), DTYPE)
        + tf.reduce_sum(tf.square(noise), axis=1)
    )
    return TailChunkResult(
        finite_z2=finite_placeholder,
        log_likelihood=log_likelihood,
        transition_log_density=transition_log_density,
        nonrepresentable_mask=nonrepresentable,
        zero_target_mask=tf.math.is_inf(log_likelihood) & (log_likelihood < 0.0),
        overflow_log_margin=tf.where(
            nonrepresentable, overflow_margin, tf.zeros_like(overflow_margin)
        ),
        ordinary_relative_residual=relative,
    )


def tail_source_closure() -> Mapping[str, str]:
    result = dict(t2_source_closure())
    for path in (
        Path(__file__).resolve().relative_to(ROOT),
        Path("scripts/prepare_zhao_cui_austria_sir_lane_b_t2_tail.py"),
        AUTHOR_SIR_STEP,
        TAIL_REPAIR_PLAN,
    ):
        result[path.as_posix()] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    return {path: result[path] for path in sorted(result)}


@dataclass(frozen=True)
class LaneBT2UntouchedTailCloud:
    reference_uniforms: tf.Tensor
    z1: tf.Tensor
    transition_noise: tf.Tensor
    previous_correction: tf.Tensor
    log_likelihood: tf.Tensor
    transition_log_density: tf.Tensor
    nonrepresentable_mask: tf.Tensor
    overflow_log_margin: tf.Tensor
    role: str
    reference_seed: int
    transition_seed: int

    def __post_init__(self) -> None:
        reference = tf.convert_to_tensor(self.reference_uniforms, DTYPE)
        if reference.shape.rank != 2 or reference.shape[0] != 18:
            raise ValueError("tail reference uniforms must have shape [18,sample]")
        sample_count = int(reference.shape[1])
        matrices = {
            "z1": tf.convert_to_tensor(self.z1, DTYPE),
            "transition_noise": tf.convert_to_tensor(self.transition_noise, DTYPE),
        }
        for name, value in matrices.items():
            if value.shape != (sample_count, 18):
                raise ValueError(f"tail {name} must have shape [sample,18]")
            tf.debugging.assert_all_finite(value, f"tail {name} must be finite")
        vectors = {
            "previous_correction": tf.convert_to_tensor(self.previous_correction, DTYPE),
            "log_likelihood": tf.convert_to_tensor(self.log_likelihood, DTYPE),
            "transition_log_density": tf.convert_to_tensor(self.transition_log_density, DTYPE),
            "overflow_log_margin": tf.convert_to_tensor(self.overflow_log_margin, DTYPE),
        }
        for name, value in vectors.items():
            if value.shape != (sample_count,):
                raise ValueError(f"tail {name} must match sample count")
            if name == "log_likelihood":
                invalid = tf.math.is_nan(value) | tf.math.is_inf(value) & (value > 0.0)
                if bool(tf.reduce_any(invalid).numpy()):
                    raise ValueError("tail likelihood contains NaN or positive infinity")
            else:
                tf.debugging.assert_all_finite(value, f"tail {name} must be finite")
        mask = tf.convert_to_tensor(self.nonrepresentable_mask, tf.bool)
        if mask.shape != (sample_count,):
            raise ValueError("tail nonrepresentable mask must match sample count")
        zero = tf.math.is_inf(vectors["log_likelihood"]) & (
            vectors["log_likelihood"] < 0.0
        )
        if bool(tf.reduce_any(mask & ~zero).numpy()):
            raise ValueError("every nonrepresentable tail row must have zero target")
        if bool(tf.reduce_any(mask & (vectors["overflow_log_margin"] <= 0.0)).numpy()):
            raise ValueError("tail overflow certificate margin must be positive")
        object.__setattr__(self, "reference_uniforms", reference)
        object.__setattr__(self, "z1", matrices["z1"])
        object.__setattr__(self, "transition_noise", matrices["transition_noise"])
        for name, value in vectors.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "nonrepresentable_mask", mask)
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "reference_seed", int(self.reference_seed))
        object.__setattr__(self, "transition_seed", int(self.transition_seed))

    @property
    def sample_count(self) -> int:
        return int(self.reference_uniforms.shape[1])

    @property
    def log_importance_weight(self) -> tf.Tensor:
        return self.previous_correction + self.log_likelihood

    @property
    def zero_target_mask(self) -> tf.Tensor:
        return tf.math.is_inf(self.log_likelihood) & (self.log_likelihood < 0.0)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "role": self.role,
            "sample_count": self.sample_count,
            "reference_seed": self.reference_seed,
            "transition_seed": self.transition_seed,
            "tail_evaluator_id": TAIL_EVALUATOR_ID,
            "reference_uniforms_sha256": tensor_sha256(self.reference_uniforms),
            "z1_sha256": tensor_sha256(self.z1),
            "transition_noise_sha256": tensor_sha256(self.transition_noise),
            "previous_correction_sha256": tensor_sha256(self.previous_correction),
            "log_likelihood_sha256": tensor_sha256(self.log_likelihood),
            "transition_log_density_sha256": tensor_sha256(self.transition_log_density),
            "nonrepresentable_mask_sha256": tensor_sha256(self.nonrepresentable_mask),
            "overflow_log_margin_sha256": tensor_sha256(self.overflow_log_margin),
            "zero_target_mask_sha256": tensor_sha256(self.zero_target_mask),
            "log_importance_weight_sha256": tensor_sha256(self.log_importance_weight),
            "nonrepresentable_count": int(
                tf.reduce_sum(tf.cast(self.nonrepresentable_mask, tf.int32)).numpy()
            ),
            "zero_target_count": int(
                tf.reduce_sum(tf.cast(self.zero_target_mask, tf.int32)).numpy()
            ),
            "all_rows_retained_in_denominator": True,
        }


def estimate_tail_log_normalizer(
    cloud: LaneBT2UntouchedTailCloud, shift_constant: tf.Tensor
) -> LaneBT2LogNormalizerEstimate:
    log_weight = cloud.log_importance_weight
    maximum = tf.reduce_max(log_weight)
    scaled = tf.exp(log_weight - maximum)
    mean_scaled = tf.reduce_mean(scaled)
    variance_scaled = tf.reduce_sum(tf.square(scaled - mean_scaled)) / tf.cast(
        cloud.sample_count - 1, DTYPE
    )
    log_increment = maximum + tf.math.log(mean_scaled)
    log_standard_error = tf.sqrt(
        variance_scaled / tf.cast(cloud.sample_count, DTYPE)
    ) / mean_scaled
    shift = tf.reshape(tf.convert_to_tensor(shift_constant, DTYPE), [])
    return LaneBT2LogNormalizerEstimate(
        role=cloud.role,
        reference_seed=cloud.reference_seed,
        transition_seed=cloud.transition_seed,
        sample_count=cloud.sample_count,
        shift_constant=shift,
        log_increment=log_increment,
        log_shifted_normalizer=shift + log_increment,
        log_standard_error=log_standard_error,
        log_importance_weight_sha256=tensor_sha256(log_weight),
    )


def load_tail_cloud(directory: Path) -> tuple[LaneBT2UntouchedTailCloud, Mapping[str, object]]:
    output = Path(directory)
    payload = json.loads((output / "result.json").read_text())
    if payload.get("schema_version") != SCHEMA or payload.get("status") != "PREPARED_T2_UNTOUCHED_TAIL_CLOUD":
        raise ValueError("T2 tail artifact schema/status mismatch")
    if payload.get("source_sha256") != dict(tail_source_closure()):
        raise ValueError("T2 tail artifact source closure mismatch")
    if not all(bool(value) for value in payload.get("gates", {}).values()):
        raise ValueError("T2 tail artifact failed a hard gate")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("T2 tail tensor ledger missing")

    def read(name: str, dtype: tf.DType) -> tf.Tensor:
        row = tensors.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"T2 tail tensor missing: {name}")
        serialized = tf.io.read_file((output / str(row["path"])).as_posix())
        if hashlib.sha256(bytes(serialized.numpy())).hexdigest() != row.get("sha256"):
            raise ValueError(f"T2 tail tensor hash mismatch: {name}")
        return tf.ensure_shape(tf.io.parse_tensor(serialized, out_type=dtype), row["shape"])

    manifest = payload["cloud_manifest"]
    cloud = LaneBT2UntouchedTailCloud(
        reference_uniforms=read("reference_uniforms", DTYPE),
        z1=read("z1", DTYPE),
        transition_noise=read("transition_noise", DTYPE),
        previous_correction=read("previous_correction", DTYPE),
        log_likelihood=read("log_likelihood", DTYPE),
        transition_log_density=read("transition_log_density", DTYPE),
        nonrepresentable_mask=read("nonrepresentable_mask", tf.bool),
        overflow_log_margin=read("overflow_log_margin", DTYPE),
        role=str(manifest["role"]),
        reference_seed=int(manifest["reference_seed"]),
        transition_seed=int(manifest["transition_seed"]),
    )
    if cloud.manifest_payload() != manifest:
        raise ValueError("T2 tail cloud manifest mismatch")
    return cloud, payload


__all__ = [
    "FP64_STANDARDIZED_RESIDUAL_LOG_THRESHOLD",
    "LaneBT2UntouchedTailCloud",
    "SCHEMA",
    "TAIL_EVALUATOR_ID",
    "TailChunkResult",
    "estimate_tail_log_normalizer",
    "evaluate_t2_tail_chunk",
    "load_tail_cloud",
    "signed_log_add",
    "signed_log_author_transition_mean",
    "signed_log_from_real",
    "signed_log_to_real",
    "tail_source_closure",
]
