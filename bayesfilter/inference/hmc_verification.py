"""Dependence-aware bounded verification contracts for HMC kernel tuning.

This module separates mean Metropolis acceptance probability from realized
accept/reject movement.  The TensorFlow telemetry helper is safe to call after
a bounded TF/TFP run; the immutable decision types are added in repair R4.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Integral
from typing import Any, Mapping


def _float64_tensor(value: Any) -> Any:
    import tensorflow as tf

    if tf.is_tensor(value):
        return tf.cast(value, tf.float64)
    return tf.convert_to_tensor(value, dtype=tf.float64)


def _all_finite(value: Any) -> bool:
    import tensorflow as tf

    return bool(tf.reduce_all(tf.math.is_finite(_float64_tensor(value))))


def _all_close(actual: Any, expected: Any, *, rtol: float, atol: float) -> bool:
    """Replay boundary with the original asymmetric reference tolerance."""

    import tensorflow as tf

    actual_tensor = _float64_tensor(actual)
    expected_tensor = _float64_tensor(expected)
    return bool(tf.reduce_all(
        tf.abs(actual_tensor - expected_tensor) <= atol + rtol * tf.abs(expected_tensor)
    ))


def _is_boolean_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    return (
        getattr(value, "shape", None) == ()
        and str(getattr(value, "dtype", None)) == "bool"
        and not hasattr(value, "__len__")
    )


ACCEPTANCE_DECISIONS = (
    "passed",
    "repair_step_lower",
    "repair_step_higher",
    "repair_trajectory",
    "inconclusive_conflict",
    "inconclusive_evidence",
    "unavailable",
)

EVIDENCE_VALIDITIES = (
    "valid",
    "candidate_data_invalid",
    "shared_execution_invalid",
)

# Exact or near-exact short-period recurrence catches deterministic cycles that
# adjacent-movement and first-to-last displacement summaries cannot detect.
_PATH_RETURN_LAGS = tuple(range(2, 17))
_PATH_RETURN_MAX_FRACTION = 0.95
_PATH_RETURN_ATOL = 1.0e-12
_PATH_RETURN_RTOL = 1.0e-10
_TARGET_HEALTH_BATCH_DRAWS = 16
_CHAIN_MEAN_UNCERTAINTY_METHOD = (
    "two_sided_student_t_independent_chain_means"
)
_LEGACY_V4_CHAIN_MEAN_UNCERTAINTY_METHOD = (
    "two_sided_hoeffding_independent_chains"
)
_STUDENT_T_CRITICAL_VALUE_90_DF3 = 2.3533634348

_SHARED_INVALIDITY_REASON_CODES = frozenset(
    {
        "nonfinite_retained_samples",
        "nonfinite_final_state",
        "required_standard_acceptance_trace_missing",
        "shared_lineage_invalid",
        "shared_adapter_invalid",
        "shared_coordinate_invalid",
        "shared_schema_invalid",
        "shared_archive_invalid",
        "shared_seed_invalid",
        "shared_callback_invalid",
        "required_target_status_telemetry_missing",
        "target_value_score_shape_invalid",
    }
)

_LEGACY_ROLE_ONLY_REASON_CODES = frozenset(
    {"log_accept_energy_proxy_exceeded", "native_divergence_positive"}
)

_CANDIDATE_DATA_INVALIDITY_REASON_CODES = frozenset(
    {
        "nonfinite_candidate_state",
        "nonfinite_log_accept_ratio",
        "nonfinite_target_log_prob",
        "nonfinite_target_score",
        "nonfinite_private_acceptance_log_value",
        "nonfinite_private_target_value",
        "native_divergence_count_missing",
        "native_divergence_provenance_inconsistent",
        "log_accept_proxy_provenance_inconsistent",
        "target_status_telemetry_failure",
        "unrecognized_health_failure",
    }
)

TARGET_STATUS_TELEMETRY_CORE_FIELDS = (
    "status_code",
    "valid_pre_regularized_score",
    "floor_count_value",
)
TARGET_STATUS_TELEMETRY_OPTIONAL_CONDITIONING_FIELDS = (
    "min_innovation_eigenvalue",
    "innovation_condition_estimate",
)
TARGET_STATUS_TELEMETRY_FIELDS = (
    *TARGET_STATUS_TELEMETRY_CORE_FIELDS,
    *TARGET_STATUS_TELEMETRY_OPTIONAL_CONDITIONING_FIELDS,
)


def _finite_tensor_mask(value: Any) -> Any:
    """Return an elementwise finite mask without moving tensor data to NumPy."""

    import tensorflow as tf

    tensor = tf.convert_to_tensor(value)
    if tensor.dtype.is_floating or tensor.dtype.is_integer:
        return tf.math.is_finite(tensor) if tensor.dtype.is_floating else tf.ones_like(
            tensor,
            dtype=tf.bool,
        )
    if tensor.dtype.is_complex:
        return tf.logical_and(
            tf.math.is_finite(tf.math.real(tensor)),
            tf.math.is_finite(tf.math.imag(tensor)),
        )
    if tensor.dtype.is_bool:
        return tf.ones_like(tensor, dtype=tf.bool)
    raise TypeError("finite tensor checks require a numeric or boolean tensor")


def target_status_telemetry_has_failure(
    telemetry: Mapping[str, Any],
    *,
    expected_shape: tuple[int, ...],
) -> bool:
    """Validate one per-chain-step target-status trace and return its veto bit."""

    if not isinstance(telemetry, Mapping):
        raise TypeError("target_status_telemetry trace must be a mapping")
    missing = tuple(
        key for key in TARGET_STATUS_TELEMETRY_CORE_FIELDS if key not in telemetry
    )
    if missing:
        raise ValueError(
            "target_status_telemetry missing required fields: " + ", ".join(missing)
        )
    optional_present = tuple(
        key in telemetry for key in TARGET_STATUS_TELEMETRY_OPTIONAL_CONDITIONING_FIELDS
    )
    if any(optional_present) and not all(optional_present):
        raise ValueError(
            "target_status_telemetry conditioning fields must be both present or both absent"
        )
    active_fields = (
        TARGET_STATUS_TELEMETRY_FIELDS
        if all(optional_present)
        else TARGET_STATUS_TELEMETRY_CORE_FIELDS
    )
    import tensorflow as tf

    tensors = {key: tf.convert_to_tensor(telemetry[key]) for key in active_fields}
    expected_tensor_shape = tf.TensorShape(expected_shape)
    if any(value.shape != expected_tensor_shape for value in tensors.values()):
        raise ValueError("target_status_telemetry fields must match the chain-step shape")
    status = tensors["status_code"]
    valid = tensors["valid_pre_regularized_score"]
    floors = tensors["floor_count_value"]
    if not status.dtype.is_integer or status.dtype.is_bool:
        raise ValueError("target status_code must be integer-valued")
    if not valid.dtype.is_bool:
        raise ValueError("target valid_pre_regularized_score must be boolean")
    if not floors.dtype.is_integer or floors.dtype.is_bool:
        raise ValueError("target floor_count_value must be integer-valued")
    status_nonvalid = (status != 0) | (~valid)
    if bool(
        tf.reduce_any(tf.logical_and(~status_nonvalid, floors < 0)).numpy()
    ):
        raise ValueError("valid target floor_count_value must be nonnegative")
    for name in TARGET_STATUS_TELEMETRY_OPTIONAL_CONDITIONING_FIELDS:
        if name not in tensors:
            continue
        value = tensors[name]
        if not value.dtype.is_numeric:
            raise ValueError(f"target {name} must be numeric")
        finite = _finite_tensor_mask(value)
        if bool(
            tf.reduce_any(tf.logical_and(~status_nonvalid, ~finite)).numpy()
        ):
            raise ValueError(f"valid target {name} must be finite")
    return bool(tf.reduce_any(status_nonvalid).numpy())


def _evaluate_retained_target_health(
    *,
    adapter: Any,
    samples: Any,
    target_status_trace_policy: str = "none",
) -> Mapping[str, Any]:
    """Recheck retained values/scores in bounded draw batches.

    TFP's standard trace exposes accepted target values but not accepted scores.
    Operational tuning calls this only after a bounded run has returned. Each
    evaluation retains the run's draw/chain axes. A non-finite chunk is
    refined only to locate its first invalid logical draw; refinement calls do
    not inflate ``evaluated_draw_count``.
    """

    policy = str(target_status_trace_policy)
    if policy not in {"none", "per_chain_step"}:
        raise ValueError(
            "target_status_trace_policy must be 'none' or 'per_chain_step'"
        )
    import tensorflow as tf

    sample_tensor = tf.cast(tf.convert_to_tensor(samples), tf.float64)
    sample_rank = sample_tensor.shape.rank
    if (
        sample_rank not in {2, 3}
        or sample_tensor.shape[0] == 0
        or sample_tensor.shape[-1] == 0
    ):
        return {
            "shared_invalidity_reasons": ("shared_schema_invalid",),
            "candidate_data_invalidity_reasons": (),
            "target_value_finite": False,
            "target_score_finite": False,
            "target_status_failure_count": None,
            "evaluated_draw_count": 0,
        }
    if not bool(tf.reduce_all(_finite_tensor_mask(sample_tensor)).numpy()):
        return {
            "shared_invalidity_reasons": ("nonfinite_retained_samples",),
            "candidate_data_invalidity_reasons": (),
            "target_value_finite": False,
            "target_score_finite": False,
            "target_status_failure_count": None,
            "evaluated_draw_count": 0,
        }
    evaluator = getattr(adapter, "log_prob_and_grad", None)
    if not callable(evaluator):
        return {
            "shared_invalidity_reasons": ("shared_adapter_invalid",),
            "candidate_data_invalidity_reasons": (),
            "target_value_finite": False,
            "target_score_finite": False,
            "target_status_failure_count": None,
            "evaluated_draw_count": 0,
        }
    telemetry = getattr(adapter, "target_status_telemetry", None)
    if policy == "per_chain_step" and not callable(telemetry):
        return {
            "shared_invalidity_reasons": (
                "required_target_status_telemetry_missing",
            ),
            "candidate_data_invalidity_reasons": (),
            "target_value_finite": False,
            "target_score_finite": False,
            "target_status_failure_count": None,
            "evaluated_draw_count": 0,
        }

    value_finite = True
    score_finite = True
    status_failure_count = 0
    evaluated = 0
    shared: list[str] = []

    supports_combined = bool(
        getattr(adapter, "supports_retained_value_score_status", False)
        and callable(getattr(adapter, "log_prob_and_grad_status", None))
    )

    def evaluate_value_score(
        values: Any,
    ) -> tuple[Any, Any, Mapping[str, Any] | None]:
        tensor = tf.cast(tf.convert_to_tensor(values), tf.float64)
        if supports_combined:
            value, score, status = adapter.log_prob_and_grad_status(tensor)
        else:
            value, score = evaluator(tensor)
            status = None
        return tf.cast(tf.convert_to_tensor(value), tf.float64), tf.cast(
            tf.convert_to_tensor(score), tf.float64
        ), status

    supports_draw_batch = bool(
        getattr(adapter, "supports_retained_draw_batch", False)
    )
    supports_flat_batch = bool(
        getattr(adapter, "supports_retained_flat_batch", False)
    )
    if supports_draw_batch and supports_flat_batch:
        raise ValueError("adapter cannot declare two retained batching contracts")
    batch_draws = (
        _TARGET_HEALTH_BATCH_DRAWS
        if supports_draw_batch or supports_flat_batch
        else 1
    )
    sample_shape = tuple(int(item) for item in sample_tensor.shape)
    for start in range(0, sample_shape[0], batch_draws):
        chunk = sample_tensor[start : start + batch_draws]
        callback_values = (
            chunk
            if supports_draw_batch
            else (
                tf.reshape(chunk, (-1, chunk.shape[-1]))
                if supports_flat_batch
                else chunk[0]
            )
        )
        expected_value_shape = tuple(int(item) for item in chunk.shape[:-1])
        callback_value_shape = (
            expected_value_shape
            if supports_draw_batch
            else tuple(int(item) for item in callback_values.shape[:-1])
        )
        try:
            value_array, score_array, status_payload = evaluate_value_score(callback_values)
        except Exception:  # noqa: BLE001 - target callback authority is broken.
            value_finite = False
            score_finite = False
            shared.append("shared_callback_invalid")
            break
        if value_array.shape != tf.TensorShape(callback_value_shape) or score_array.shape != callback_values.shape:
            shared.append("target_value_score_shape_invalid")
            break
        if supports_flat_batch:
            value_array = tf.reshape(value_array, expected_value_shape)
            score_array = tf.reshape(score_array, chunk.shape)
        chunk_value_finite = bool(tf.reduce_all(_finite_tensor_mask(value_array)).numpy())
        chunk_score_finite = bool(tf.reduce_all(_finite_tensor_mask(score_array)).numpy())
        if not chunk_value_finite or not chunk_score_finite:
            # Refine only the first failing chunk. Count unique logical draws,
            # not the diagnostic callback invocations used for localization.
            for offset in range(int(chunk.shape[0])):
                draw = chunk[offset]
                draw_expected_shape = tuple(int(item) for item in draw.shape[:-1])
                try:
                    draw_value, draw_score, _draw_status = evaluate_value_score(draw)
                except Exception:  # noqa: BLE001 - callback authority is broken.
                    evaluated += offset
                    value_finite = False
                    score_finite = False
                    shared.append("shared_callback_invalid")
                    break
                if (
                    draw_value.shape != draw_expected_shape
                    or draw_score.shape != draw.shape
                ):
                    evaluated += offset
                    shared.append("target_value_score_shape_invalid")
                    break
                draw_value_finite = bool(
                    tf.reduce_all(_finite_tensor_mask(draw_value)).numpy()
                )
                draw_score_finite = bool(
                    tf.reduce_all(_finite_tensor_mask(draw_score)).numpy()
                )
                if not draw_value_finite or not draw_score_finite:
                    evaluated += offset + 1
                    value_finite = draw_value_finite
                    score_finite = draw_score_finite
                    break
            else:
                # A failure that exists only in the batched callback cannot be
                # assigned to a logical draw under the declared contract.
                shared.append("shared_schema_invalid")
            break
        if policy == "per_chain_step":
            try:
                if not supports_combined:
                    status_payload = telemetry(
                        tf.convert_to_tensor(callback_values, dtype=tf.float64)
                    )
                if not isinstance(status_payload, Mapping):
                    raise TypeError("combined target status must be a mapping")
            except Exception:  # noqa: BLE001 - target callback authority is broken.
                shared.append("shared_callback_invalid")
                break
            try:
                if target_status_telemetry_has_failure(
                    status_payload,
                    expected_shape=callback_value_shape,
                ):
                    if batch_draws == 1:
                        status = tf.convert_to_tensor(status_payload["status_code"])
                        valid = tf.convert_to_tensor(
                            status_payload["valid_pre_regularized_score"],
                            dtype=tf.bool,
                        )
                        status_failure_count += int(
                            tf.reduce_sum(
                                tf.cast((status != 0) | (~valid), tf.int32)
                            ).numpy()
                        )
                        evaluated += int(chunk.shape[0])
                        # Keep scanning the retained draws so the public
                        # count reflects all logical failures, not only the
                        # first one encountered.
                        continue
                    draw_failure_found = False
                    for offset, draw in enumerate(chunk):
                        draw_expected_shape = tuple(
                            int(item) for item in draw.shape[:-1]
                        )
                        try:
                            if supports_combined:
                                _draw_value, _draw_score, draw_payload = (
                                    evaluate_value_score(draw)
                                )
                            else:
                                draw_payload = telemetry(
                                    tf.convert_to_tensor(draw, dtype=tf.float64)
                                )
                        except Exception:  # noqa: BLE001 - callback authority is broken.
                            evaluated += offset
                            shared.append("shared_callback_invalid")
                            break
                        try:
                            draw_failed = target_status_telemetry_has_failure(
                                draw_payload,
                                expected_shape=draw_expected_shape,
                            )
                        except (AttributeError, TypeError, ValueError):
                            evaluated += offset
                            shared.append("shared_schema_invalid")
                            break
                        if draw_failed:
                            draw_failure_found = True
                            status = tf.convert_to_tensor(
                                draw_payload["status_code"]
                            )
                            valid = tf.convert_to_tensor(
                                draw_payload["valid_pre_regularized_score"],
                                dtype=tf.bool,
                            )
                            status_failure_count += int(
                                tf.reduce_sum(
                                    tf.cast((status != 0) | (~valid), tf.int32)
                                ).numpy()
                            )
                            continue
                    if shared:
                        break
                    if not draw_failure_found and not shared:
                        # A batched-only failure that cannot be reproduced per
                        # logical draw violates the declared adapter contract.
                        shared.append("shared_schema_invalid")
                        break
                    # A valid telemetry failure is candidate-local evidence;
                    # continue scanning later chunks to count every failure.
                    evaluated += int(chunk.shape[0])
                    continue
            except (AttributeError, TypeError, ValueError):
                shared.append("shared_schema_invalid")
                break
        evaluated += int(chunk.shape[0])

    candidate: list[str] = []
    if not shared:
        if not value_finite:
            candidate.append("nonfinite_target_log_prob")
        if not score_finite:
            candidate.append("nonfinite_target_score")
        if status_failure_count:
            candidate.append("target_status_telemetry_failure")
    return {
        "shared_invalidity_reasons": tuple(dict.fromkeys(shared)),
        "candidate_data_invalidity_reasons": tuple(dict.fromkeys(candidate)),
        "target_value_finite": bool(value_finite and not shared),
        "target_score_finite": bool(score_finite and not shared),
        "target_status_failure_count": (
            None if policy == "none" or shared else int(status_failure_count)
        ),
        "evaluated_draw_count": int(evaluated),
    }


def _strict_scalar_integer(value: Any, *, name: str) -> int:
    """Return an actual scalar integer without truncating malformed payloads."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer scalar")
    return int(value)


@dataclass(frozen=True)
class HMCAcceptancePolicy:
    """Dependence-aware fixed-checkpoint policy for bounded HMC verification."""

    target: float = 0.70
    practical_region: tuple[float, float] = (0.65, 0.75)
    repair_region: tuple[float, float] = (0.55, 0.85)
    chain_count: int = 4
    block_count: int = 4
    min_block_size: int = 16
    confidence_level: float = 0.90
    min_movement_rate: float = 0.05
    max_repeated_state_fraction: float = 0.95
    min_normalized_return_displacement: float = 1.0e-4
    max_abs_log_accept_energy_proxy: float = 1000.0
    allowed_cost_stop_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        target = float(self.target)
        practical = tuple(float(item) for item in self.practical_region)
        repair = tuple(float(item) for item in self.repair_region)
        if not math.isfinite(target):
            raise ValueError("target must be finite")
        if len(practical) != 2 or not 0.0 < practical[0] < practical[1] < 1.0:
            raise ValueError("practical_region must be ordered inside (0, 1)")
        if len(repair) != 2 or not 0.0 < repair[0] < repair[1] < 1.0:
            raise ValueError("repair_region must be ordered inside (0, 1)")
        if not repair[0] <= practical[0] <= target <= practical[1] <= repair[1]:
            raise ValueError(
                "target must lie in practical_region and practical_region in repair_region"
            )
        if not (
            abs(target - 0.5 * sum(practical)) <= 1.0e-12
            and abs(target - 0.5 * sum(repair)) <= 1.0e-12
        ):
            raise ValueError(
                "practical_region and repair_region must be centered on target"
            )
        chain_count = _strict_scalar_integer(self.chain_count, name="chain_count")
        block_count = _strict_scalar_integer(self.block_count, name="block_count")
        min_block_size = _strict_scalar_integer(
            self.min_block_size,
            name="min_block_size",
        )
        if chain_count != 4:
            raise ValueError("R0-R8 acceptance policy requires exactly four chains")
        if block_count != 4 or min_block_size < 16:
            raise ValueError("acceptance policy requires four blocks of at least 16")
        if float(self.confidence_level) != 0.90:
            raise ValueError("R0-R8 acceptance policy supports only a 90% interval")
        for name in ("min_movement_rate", "max_repeated_state_fraction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and inside [0, 1]")
            object.__setattr__(self, name, value)
        for name in (
            "min_normalized_return_displacement",
            "max_abs_log_accept_energy_proxy",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "practical_region", practical)
        object.__setattr__(self, "repair_region", repair)
        object.__setattr__(self, "chain_count", chain_count)
        object.__setattr__(self, "block_count", block_count)
        object.__setattr__(self, "min_block_size", min_block_size)
        cost_stop_reasons = tuple(
            dict.fromkeys(str(item) for item in self.allowed_cost_stop_reasons)
        )
        if any(not item for item in cost_stop_reasons):
            raise ValueError("allowed_cost_stop_reasons must contain nonempty codes")
        object.__setattr__(self, "allowed_cost_stop_reasons", cost_stop_reasons)

    @property
    def min_decisions_per_chain(self) -> int:
        return self.block_count * self.min_block_size

    @property
    def student_critical_value(self) -> float:
        """Two-sided 90% Student-t critical value for four chain means."""

        return _STUDENT_T_CRITICAL_VALUE_90_DF3

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_acceptance_policy.v5",
            "target": self.target,
            "practical_region": self.practical_region,
            "repair_region": self.repair_region,
            "chain_count": self.chain_count,
            "block_count": self.block_count,
            "min_block_size": self.min_block_size,
            "min_decisions_per_chain": self.min_decisions_per_chain,
            "confidence_level": self.confidence_level,
            "min_movement_rate": self.min_movement_rate,
            "max_repeated_state_fraction": self.max_repeated_state_fraction,
            "min_normalized_return_displacement": (
                self.min_normalized_return_displacement
            ),
            "path_return_contract": {
                "lags": _PATH_RETURN_LAGS,
                "aggregation": "maximum_recurrence_fraction_over_lags_per_chain",
                "minimum_repetitions_at_minimum_evidence": 4,
                "max_fraction": _PATH_RETURN_MAX_FRACTION,
                "absolute_tolerance": _PATH_RETURN_ATOL,
                "relative_tolerance": _PATH_RETURN_RTOL,
                "chain_rule": "veto_if_any_chain_exceeds_max_fraction",
            },
            "max_abs_log_accept_energy_proxy": (
                self.max_abs_log_accept_energy_proxy
            ),
            "allowed_cost_stop_reasons": self.allowed_cost_stop_reasons,
            "dependence_unit": "independently_seeded_chain_mean",
            "uncertainty_method": _CHAIN_MEAN_UNCERTAINTY_METHOD,
            "tuning_decision_role": (
                "working_tuning_compatibility_interval_not_convergence_or_equivalence"
            ),
            "diagnostic_roles": {
                "mean_acceptance_probability": "promotion_criterion_and_repair_trigger",
                "movement": "promotion_veto_and_trajectory_repair_trigger",
                "bounded_short_cycle_path_return": (
                    "promotion_veto_and_resonance_repair_trigger"
                ),
                "native_divergence": "promotion_veto",
                "max_abs_log_accept_energy_proxy": "explanatory_alert_only",
                "signed_log_accept_ratio_tails": "explanatory_alert_only",
            },
        }


@dataclass(frozen=True)
class HMCAcceptanceEvidence:
    """Role-separated evidence for one fixed HMC verification checkpoint."""

    evidence_validity: str
    acceptance_decision: str
    pooled_mean: float | None
    chain_mean_uncertainty_interval: tuple[float, float] | None
    chain_mean_uncertainty_method: str | None
    chain_mean_uncertainty_level: float | None
    chain_means: tuple[float, ...]
    block_means_by_chain: tuple[tuple[float, ...], ...]
    realized_acceptance_rate: float | None
    realized_acceptance_rate_by_chain: tuple[float, ...]
    movement_rate_by_chain: tuple[float, ...]
    repeated_state_fraction_by_chain: tuple[float, ...]
    normalized_return_displacement_by_chain: tuple[float, ...]
    path_return_fraction_by_chain: tuple[float, ...]
    usable_decisions_per_chain: int
    excluded_remainder_per_chain: int
    native_divergence_status: str
    native_divergence_count: int | None
    min_log_accept_ratio: float | None
    max_log_accept_ratio: float | None
    max_abs_log_accept_energy_proxy: float | None
    negative_proxy_exceedance_count_by_chain: tuple[int, ...]
    positive_proxy_exceedance_count_by_chain: tuple[int, ...]
    negative_proxy_exceedance_rate_by_chain: tuple[float, ...]
    positive_proxy_exceedance_rate_by_chain: tuple[float, ...]
    policy: HMCAcceptancePolicy
    engineering_invalidity_reasons: tuple[str, ...] = ()
    candidate_promotion_vetoes: tuple[str, ...] = ()
    tuning_repair_triggers: tuple[str, ...] = ()
    candidate_health_alerts: tuple[str, ...] = ()
    diagnostic_followups: tuple[str, ...] = ()
    cost_stop_reasons: tuple[str, ...] = ()
    explanatory_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.policy, HMCAcceptancePolicy):
            raise TypeError("policy must be HMCAcceptancePolicy")
        validity = str(self.evidence_validity)
        if validity not in EVIDENCE_VALIDITIES:
            raise ValueError("unsupported HMC evidence validity")
        decision = str(self.acceptance_decision)
        if decision not in ACCEPTANCE_DECISIONS:
            raise ValueError("unsupported HMC acceptance decision")
        status = str(self.native_divergence_status)
        if status not in {"available", "not_exposed_by_kernel", "not_collected"}:
            raise ValueError("invalid native divergence status")
        if self.native_divergence_count is None:
            divergence_count = None
        elif isinstance(self.native_divergence_count, Integral) and not isinstance(
            self.native_divergence_count, bool
        ):
            divergence_count = int(self.native_divergence_count)
        else:
            raise ValueError("native_divergence_count must be an integer when provided")
        if divergence_count is not None and divergence_count < 0:
            raise ValueError("native_divergence_count must be nonnegative")
        if status == "available" and divergence_count is None:
            raise ValueError("available divergence status requires a count")
        if status != "available" and divergence_count is not None:
            raise ValueError("unavailable divergence status cannot carry a count")
        object.__setattr__(self, "native_divergence_status", status)
        object.__setattr__(self, "native_divergence_count", divergence_count)

        role_names = (
            "engineering_invalidity_reasons",
            "candidate_promotion_vetoes",
            "tuning_repair_triggers",
            "candidate_health_alerts",
            "diagnostic_followups",
            "cost_stop_reasons",
            "explanatory_notes",
        )
        for name in role_names:
            values = tuple(dict.fromkeys(str(item) for item in getattr(self, name)))
            if any(not item for item in values):
                raise ValueError(f"{name} must contain nonempty reason codes")
            object.__setattr__(self, name, values)
        invalidity = self.engineering_invalidity_reasons
        if validity == "valid" and invalidity:
            raise ValueError("valid evidence cannot carry engineering invalidity")
        if validity != "valid" and not invalidity:
            raise ValueError("invalid evidence requires an engineering invalidity reason")
        if validity != "valid" and decision != "unavailable":
            raise ValueError("invalid evidence must mark acceptance unavailable")
        if validity == "valid" and decision == "unavailable":
            raise ValueError("finite valid v5 evidence cannot mark acceptance unavailable")
        allowed_invalidity = (
            _SHARED_INVALIDITY_REASON_CODES
            if validity == "shared_execution_invalid"
            else _CANDIDATE_DATA_INVALIDITY_REASON_CODES
        ) | {"unrecognized_health_failure"}
        if invalidity and not set(invalidity).issubset(allowed_invalidity):
            raise ValueError("engineering invalidity contains an unsupported reason code")

        interval = None
        if self.chain_mean_uncertainty_interval is not None:
            interval = tuple(float(item) for item in self.chain_mean_uncertainty_interval)
            if len(interval) != 2 or not all(math.isfinite(item) for item in interval):
                raise ValueError("uncertainty interval must contain two finite values")
            if interval[0] > interval[1] or interval[0] < 0.0 or interval[1] > 1.0:
                raise ValueError("uncertainty interval must be ordered inside [0, 1]")
        object.__setattr__(self, "chain_mean_uncertainty_interval", interval)
        method = self.chain_mean_uncertainty_method
        level = self.chain_mean_uncertainty_level
        if interval is None:
            if method is not None or level is not None:
                raise ValueError("missing uncertainty interval cannot carry its metadata")
        else:
            if method != _CHAIN_MEAN_UNCERTAINTY_METHOD:
                raise ValueError("unsupported chain-mean uncertainty method")
            level = float(level) if level is not None else float("nan")
            if not math.isfinite(level) or not 0.0 < level < 1.0:
                raise ValueError("uncertainty level must be inside (0, 1)")
        object.__setattr__(self, "chain_mean_uncertainty_method", method)
        object.__setattr__(self, "chain_mean_uncertainty_level", level)
        if self.pooled_mean is not None:
            pooled_mean = float(self.pooled_mean)
            if not math.isfinite(pooled_mean):
                raise ValueError("pooled_mean must be finite when provided")
            object.__setattr__(self, "pooled_mean", pooled_mean)
        if self.pooled_mean is not None and not 0.0 <= self.pooled_mean <= 1.0:
            raise ValueError("pooled_mean must lie inside [0, 1]")
        for name in (
            "chain_means",
            "realized_acceptance_rate_by_chain",
            "movement_rate_by_chain",
            "repeated_state_fraction_by_chain",
            "normalized_return_displacement_by_chain",
            "path_return_fraction_by_chain",
        ):
            values = tuple(float(item) for item in getattr(self, name))
            if not all(math.isfinite(item) for item in values):
                raise ValueError(f"{name} must contain only finite values")
            object.__setattr__(self, name, values)
        if any(not 0.0 <= item <= 1.0 for item in self.chain_means):
            raise ValueError("chain_means must lie inside [0, 1]")
        for name in (
            "realized_acceptance_rate_by_chain",
            "movement_rate_by_chain",
            "repeated_state_fraction_by_chain",
            "path_return_fraction_by_chain",
        ):
            if any(not 0.0 <= item <= 1.0 for item in getattr(self, name)):
                raise ValueError(f"{name} must lie inside [0, 1]")
        realized_rate = self.realized_acceptance_rate
        if realized_rate is not None:
            realized_rate = float(realized_rate)
            if not math.isfinite(realized_rate) or not 0.0 <= realized_rate <= 1.0:
                raise ValueError("realized_acceptance_rate must lie inside [0, 1]")
        object.__setattr__(self, "realized_acceptance_rate", realized_rate)
        if any(item < 0.0 for item in self.normalized_return_displacement_by_chain):
            raise ValueError("normalized_return_displacement_by_chain must be nonnegative")
        block_means = tuple(
            tuple(float(item) for item in row) for row in self.block_means_by_chain
        )
        if any(not all(math.isfinite(item) for item in row) for row in block_means):
            raise ValueError("block_means_by_chain must contain only finite values")
        if any(any(not 0.0 <= item <= 1.0 for item in row) for row in block_means):
            raise ValueError("block_means_by_chain must lie inside [0, 1]")
        object.__setattr__(self, "block_means_by_chain", block_means)

        usable = _strict_scalar_integer(
            self.usable_decisions_per_chain,
            name="usable_decisions_per_chain",
        )
        excluded = _strict_scalar_integer(
            self.excluded_remainder_per_chain,
            name="excluded_remainder_per_chain",
        )
        if usable < 0 or excluded < 0:
            raise ValueError("decision counts must be nonnegative")
        object.__setattr__(self, "usable_decisions_per_chain", usable)
        object.__setattr__(self, "excluded_remainder_per_chain", excluded)
        _validate_signed_proxy_summary(self)

        if validity != "valid":
            if any(
                (
                    self.pooled_mean is not None,
                    interval is not None,
                    bool(self.chain_means),
                    bool(block_means),
                    self.realized_acceptance_rate is not None,
                    bool(self.realized_acceptance_rate_by_chain),
                    bool(self.movement_rate_by_chain),
                    bool(self.repeated_state_fraction_by_chain),
                    bool(self.normalized_return_displacement_by_chain),
                    bool(self.path_return_fraction_by_chain),
                    usable != 0,
                    excluded != 0,
                    self.min_log_accept_ratio is not None,
                    self.max_log_accept_ratio is not None,
                    self.max_abs_log_accept_energy_proxy is not None,
                    bool(self.negative_proxy_exceedance_count_by_chain),
                    bool(self.positive_proxy_exceedance_count_by_chain),
                )
            ):
                raise ValueError("invalid evidence cannot carry acceptance summaries")
            if any(
                (
                    bool(self.candidate_promotion_vetoes),
                    bool(self.tuning_repair_triggers),
                    bool(self.candidate_health_alerts),
                    bool(self.diagnostic_followups),
                    bool(self.cost_stop_reasons),
                )
            ):
                raise ValueError("invalid evidence cannot carry candidate action roles")
        else:
            _validate_valid_acceptance_summary(self, block_means)
            _validate_v5_roles(self)
        object.__setattr__(self, "evidence_validity", validity)
        object.__setattr__(self, "acceptance_decision", decision)

    @property
    def decision(self) -> str:
        """Deprecated acceptance-only alias; never encodes validity or promotion."""

        return self.acceptance_decision

    @property
    def interval(self) -> tuple[float, float] | None:
        """Deprecated alias for the v5 working chain-mean interval."""

        return self.chain_mean_uncertainty_interval

    @property
    def standard_error(self) -> None:
        """The v3 pseudo-standard-error has no v5 statistical meaning."""

        return None

    @property
    def hard_health_failures(self) -> tuple[str, ...]:
        """Deprecated display alias for engineering invalidity only."""

        return self.engineering_invalidity_reasons

    @property
    def promotion_eligible(self) -> bool:
        return (
            self.evidence_validity == "valid"
            and self.acceptance_decision == "passed"
            and not self.candidate_promotion_vetoes
            and not self.cost_stop_reasons
        )

    @property
    def cost_stop_scope(self) -> str | None:
        return "exact_candidate_replication" if self.cost_stop_reasons else None

    @property
    def passed(self) -> bool:
        """Compatibility flag with safe v5 promotion semantics."""

        return self.promotion_eligible

    @property
    def repair_direction(self) -> str | None:
        trigger = {
            "repair_step_lower": "step_size:lower_epsilon",
            "repair_step_higher": "step_size:higher_epsilon",
        }.get(self.acceptance_decision)
        if trigger is None or trigger not in self.tuning_repair_triggers:
            return None
        return trigger.split(":", 1)[1]

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_acceptance_evidence.v5",
            "evidence_validity": self.evidence_validity,
            "acceptance_decision": self.acceptance_decision,
            "decision": self.acceptance_decision,
            "passed": self.promotion_eligible,
            "promotion_eligible": self.promotion_eligible,
            "repair_direction": self.repair_direction,
            "pooled_mean": self.pooled_mean,
            "chain_mean_uncertainty_interval": self.chain_mean_uncertainty_interval,
            "chain_mean_uncertainty_method": self.chain_mean_uncertainty_method,
            "chain_mean_uncertainty_level": self.chain_mean_uncertainty_level,
            "chain_means": self.chain_means,
            "block_means_by_chain": self.block_means_by_chain,
            "realized_acceptance_rate": self.realized_acceptance_rate,
            "realized_acceptance_rate_by_chain": self.realized_acceptance_rate_by_chain,
            "movement_rate_by_chain": self.movement_rate_by_chain,
            "repeated_state_fraction_by_chain": self.repeated_state_fraction_by_chain,
            "normalized_return_displacement_by_chain": self.normalized_return_displacement_by_chain,
            "path_return_fraction_by_chain": self.path_return_fraction_by_chain,
            "usable_decisions_per_chain": self.usable_decisions_per_chain,
            "excluded_remainder_per_chain": self.excluded_remainder_per_chain,
            "native_divergence_status": self.native_divergence_status,
            "native_divergence_count": self.native_divergence_count,
            "min_log_accept_ratio": self.min_log_accept_ratio,
            "max_log_accept_ratio": self.max_log_accept_ratio,
            "max_abs_log_accept_energy_proxy": self.max_abs_log_accept_energy_proxy,
            "negative_proxy_exceedance_count_by_chain": self.negative_proxy_exceedance_count_by_chain,
            "positive_proxy_exceedance_count_by_chain": self.positive_proxy_exceedance_count_by_chain,
            "negative_proxy_exceedance_rate_by_chain": self.negative_proxy_exceedance_rate_by_chain,
            "positive_proxy_exceedance_rate_by_chain": self.positive_proxy_exceedance_rate_by_chain,
            "policy": self.policy.payload(),
            "engineering_invalidity_reasons": self.engineering_invalidity_reasons,
            "candidate_promotion_vetoes": self.candidate_promotion_vetoes,
            "tuning_repair_triggers": self.tuning_repair_triggers,
            "candidate_health_alerts": self.candidate_health_alerts,
            "diagnostic_followups": self.diagnostic_followups,
            "cost_stop_reasons": self.cost_stop_reasons,
            "cost_stop_scope": self.cost_stop_scope,
            "explanatory_notes": self.explanatory_notes,
            "raw_traces_exposed": False,
            "reports_posterior_convergence": False,
            "compatibility_aliases_non_authoritative": True,
        }


def hmc_acceptance_evidence_from_payload(
    payload: Mapping[str, Any],
) -> HMCAcceptanceEvidence:
    """Reconstruct and fully validate one live v5 acceptance-evidence payload."""

    if not isinstance(payload, Mapping):
        raise TypeError("acceptance evidence payload must be a mapping")
    if payload.get("schema") != "bayesfilter.hmc_acceptance_evidence.v5":
        raise ValueError("acceptance evidence schema mismatch")
    expected_keys = {
        "schema",
        "evidence_validity",
        "acceptance_decision",
        "decision",
        "passed",
        "promotion_eligible",
        "repair_direction",
        "pooled_mean",
        "chain_mean_uncertainty_interval",
        "chain_mean_uncertainty_method",
        "chain_mean_uncertainty_level",
        "chain_means",
        "block_means_by_chain",
        "realized_acceptance_rate",
        "realized_acceptance_rate_by_chain",
        "movement_rate_by_chain",
        "repeated_state_fraction_by_chain",
        "normalized_return_displacement_by_chain",
        "path_return_fraction_by_chain",
        "usable_decisions_per_chain",
        "excluded_remainder_per_chain",
        "native_divergence_status",
        "native_divergence_count",
        "min_log_accept_ratio",
        "max_log_accept_ratio",
        "max_abs_log_accept_energy_proxy",
        "negative_proxy_exceedance_count_by_chain",
        "positive_proxy_exceedance_count_by_chain",
        "negative_proxy_exceedance_rate_by_chain",
        "positive_proxy_exceedance_rate_by_chain",
        "policy",
        "engineering_invalidity_reasons",
        "candidate_promotion_vetoes",
        "tuning_repair_triggers",
        "candidate_health_alerts",
        "diagnostic_followups",
        "cost_stop_reasons",
        "cost_stop_scope",
        "explanatory_notes",
        "raw_traces_exposed",
        "reports_posterior_convergence",
        "compatibility_aliases_non_authoritative",
    }
    if set(payload) != expected_keys:
        raise ValueError("acceptance evidence field set is inconsistent")
    policy = _acceptance_policy_from_payload(payload.get("policy"))
    evidence = HMCAcceptanceEvidence(
        evidence_validity=str(payload.get("evidence_validity", "")),
        acceptance_decision=str(payload.get("acceptance_decision", "")),
        pooled_mean=payload.get("pooled_mean"),
        chain_mean_uncertainty_interval=(
            None
            if payload.get("chain_mean_uncertainty_interval") is None
            else tuple(payload["chain_mean_uncertainty_interval"])
        ),
        chain_mean_uncertainty_method=payload.get(
            "chain_mean_uncertainty_method"
        ),
        chain_mean_uncertainty_level=payload.get("chain_mean_uncertainty_level"),
        chain_means=tuple(payload.get("chain_means", ())),
        block_means_by_chain=tuple(tuple(row) for row in payload.get("block_means_by_chain", ())),
        realized_acceptance_rate=payload.get("realized_acceptance_rate"),
        realized_acceptance_rate_by_chain=tuple(payload.get("realized_acceptance_rate_by_chain", ())),
        movement_rate_by_chain=tuple(payload.get("movement_rate_by_chain", ())),
        repeated_state_fraction_by_chain=tuple(payload.get("repeated_state_fraction_by_chain", ())),
        normalized_return_displacement_by_chain=tuple(payload.get("normalized_return_displacement_by_chain", ())),
        path_return_fraction_by_chain=tuple(payload.get("path_return_fraction_by_chain", ())),
        usable_decisions_per_chain=payload.get("usable_decisions_per_chain", 0),
        excluded_remainder_per_chain=payload.get("excluded_remainder_per_chain", 0),
        native_divergence_status=str(payload.get("native_divergence_status", "")),
        native_divergence_count=payload.get("native_divergence_count"),
        min_log_accept_ratio=payload.get("min_log_accept_ratio"),
        max_log_accept_ratio=payload.get("max_log_accept_ratio"),
        max_abs_log_accept_energy_proxy=payload.get("max_abs_log_accept_energy_proxy"),
        negative_proxy_exceedance_count_by_chain=tuple(payload.get("negative_proxy_exceedance_count_by_chain", ())),
        positive_proxy_exceedance_count_by_chain=tuple(payload.get("positive_proxy_exceedance_count_by_chain", ())),
        negative_proxy_exceedance_rate_by_chain=tuple(payload.get("negative_proxy_exceedance_rate_by_chain", ())),
        positive_proxy_exceedance_rate_by_chain=tuple(payload.get("positive_proxy_exceedance_rate_by_chain", ())),
        policy=policy,
        engineering_invalidity_reasons=tuple(payload.get("engineering_invalidity_reasons", ())),
        candidate_promotion_vetoes=tuple(payload.get("candidate_promotion_vetoes", ())),
        tuning_repair_triggers=tuple(payload.get("tuning_repair_triggers", ())),
        candidate_health_alerts=tuple(payload.get("candidate_health_alerts", ())),
        diagnostic_followups=tuple(payload.get("diagnostic_followups", ())),
        cost_stop_reasons=tuple(payload.get("cost_stop_reasons", ())),
        explanatory_notes=tuple(payload.get("explanatory_notes", ())),
    )
    if payload.get("decision") != evidence.acceptance_decision:
        raise ValueError("acceptance evidence decision is inconsistent")
    if payload.get("passed") is not evidence.promotion_eligible:
        raise ValueError("acceptance evidence passed flag is inconsistent")
    if payload.get("promotion_eligible") is not evidence.promotion_eligible:
        raise ValueError("acceptance evidence promotion eligibility is inconsistent")
    if payload.get("repair_direction") != evidence.repair_direction:
        raise ValueError("acceptance evidence repair direction is inconsistent")
    if payload.get("cost_stop_scope") != evidence.cost_stop_scope:
        raise ValueError("acceptance evidence cost-stop scope is inconsistent")
    if payload.get("raw_traces_exposed") is not False:
        raise ValueError("acceptance evidence must not expose raw traces")
    if payload.get("reports_posterior_convergence") is not False:
        raise ValueError("acceptance evidence cannot report posterior convergence")
    if payload.get("compatibility_aliases_non_authoritative") is not True:
        raise ValueError("acceptance evidence compatibility marker is inconsistent")
    return evidence


def hmc_acceptance_evidence_v3_migration_view(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Expose historical v3 evidence without granting v4 promotion authority."""

    if not isinstance(payload, Mapping):
        raise TypeError("legacy v3 acceptance evidence must be a mapping")
    if payload.get("schema") != "bayesfilter.hmc_acceptance_evidence.v3":
        raise ValueError("legacy v3 acceptance evidence schema mismatch")
    required = {
        "acceptance_decision",
        "promotion_eligible",
        "interval",
        "standard_error",
    }
    if not required.issubset(payload):
        raise ValueError("legacy v3 acceptance evidence is incomplete")
    return {
        "schema": "bayesfilter.hmc_acceptance_evidence_v3_migration_view.v1",
        "source_schema": "bayesfilter.hmc_acceptance_evidence.v3",
        "acceptance_decision": "recompute_required",
        "promotion_eligible": False,
        "promotion_eligible_under_v4": False,
        "historical_v3_acceptance_decision": payload["acceptance_decision"],
        "historical_v3_promotion_eligible": payload["promotion_eligible"],
        "historical_v3_interval": payload["interval"],
        "historical_v3_standard_error": payload["standard_error"],
        "fresh_raw_trace_required": True,
        "source_payload_mutated": False,
    }


def _validate_hmc_acceptance_evidence_v4_payload(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the historical v4 envelope without applying v5 semantics."""

    if not isinstance(payload, Mapping):
        raise TypeError("legacy v4 acceptance evidence must be a mapping")
    if payload.get("schema") != "bayesfilter.hmc_acceptance_evidence.v4":
        raise ValueError("legacy v4 acceptance evidence schema mismatch")
    required = {
        "acceptance_decision",
        "decision",
        "passed",
        "promotion_eligible",
        "pooled_mean",
        "chain_mean_uncertainty_interval",
        "chain_mean_uncertainty_method",
        "policy",
        "raw_traces_exposed",
        "reports_posterior_convergence",
    }
    if not required.issubset(payload):
        raise ValueError("legacy v4 acceptance evidence is incomplete")
    decision = str(payload["acceptance_decision"])
    if decision not in ACCEPTANCE_DECISIONS:
        raise ValueError("legacy v4 acceptance decision is unsupported")
    if payload.get("decision") != decision:
        raise ValueError("legacy v4 acceptance decision alias is inconsistent")
    if not _is_boolean_scalar(payload.get("passed")):
        raise ValueError("legacy v4 acceptance passed flag is malformed")
    if not _is_boolean_scalar(payload.get("promotion_eligible")):
        raise ValueError("legacy v4 promotion flag is malformed")
    if payload.get("passed") is not payload.get("promotion_eligible"):
        raise ValueError("legacy v4 promotion flags are inconsistent")
    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("legacy v4 acceptance policy is incomplete")
    if policy.get("schema") != "bayesfilter.hmc_acceptance_policy.v4":
        raise ValueError("legacy v4 acceptance policy schema mismatch")
    if policy.get("uncertainty_method") != _LEGACY_V4_CHAIN_MEAN_UNCERTAINTY_METHOD:
        raise ValueError("legacy v4 uncertainty method is inconsistent")
    interval = payload.get("chain_mean_uncertainty_interval")
    if interval is not None:
        values = tuple(float(item) for item in interval)
        if len(values) != 2 or not all(math.isfinite(item) for item in values):
            raise ValueError("legacy v4 uncertainty interval is malformed")
        if values[0] > values[1] or values[0] < 0.0 or values[1] > 1.0:
            raise ValueError("legacy v4 uncertainty interval is malformed")
    if payload.get("chain_mean_uncertainty_method") != _LEGACY_V4_CHAIN_MEAN_UNCERTAINTY_METHOD:
        raise ValueError("legacy v4 evidence uncertainty method is inconsistent")
    if payload.get("raw_traces_exposed") is not False:
        raise ValueError("legacy v4 acceptance evidence exposes raw traces")
    if payload.get("reports_posterior_convergence") is not False:
        raise ValueError("legacy v4 acceptance evidence claims posterior convergence")
    return payload


def hmc_acceptance_evidence_v4_migration_view(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Expose v4 evidence diagnostically while forbidding v5 promotion.

    v4 used a fixed Hoeffding-width summary and target-centered point gates.
    Without raw acceptance traces, its decision cannot be recomputed under the
    v5 Student-t interval contract; even a historically passing v4 payload is
    therefore explicitly non-promoting.
    """

    legacy = _validate_hmc_acceptance_evidence_v4_payload(payload)
    return {
        "schema": "bayesfilter.hmc_acceptance_evidence_v4_migration_view.v1",
        "source_schema": "bayesfilter.hmc_acceptance_evidence.v4",
        "source_policy_schema": "bayesfilter.hmc_acceptance_policy.v4",
        "acceptance_decision": "recompute_required",
        "promotion_eligible": False,
        "promotion_eligible_under_v5": False,
        "historical_v4_acceptance_decision": legacy["acceptance_decision"],
        "historical_v4_promotion_eligible": bool(legacy["promotion_eligible"]),
        "historical_v4_interval": legacy["chain_mean_uncertainty_interval"],
        "historical_v4_uncertainty_method": legacy[
            "chain_mean_uncertainty_method"
        ],
        "fresh_raw_trace_required": True,
        "source_payload_mutated": False,
    }


def hmc_acceptance_evidence_v2_migration_view(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate a legacy v2 payload and expose a non-actionable migration view."""

    legacy = _validate_hmc_acceptance_evidence_v2_payload(payload)
    failures = tuple(str(item) for item in legacy.get("hard_health_failures", ()))
    proxy_alert = "log_accept_energy_proxy_exceeded" in failures
    missing_acceptance = legacy.get("pooled_mean") is None
    return {
        "schema": "bayesfilter.hmc_acceptance_evidence_v2_migration_view.v1",
        "source_schema": "bayesfilter.hmc_acceptance_evidence.v2",
        "historical_contract_validity": "valid_under_historical_contract",
        "historical_decision": legacy["decision"],
        "v3_reanalysis_status": (
            "impossible_missing_raw_acceptance_trace"
            if missing_acceptance
            else "requires_new_raw_trace_reanalysis"
        ),
        "promotion_eligible_under_v3": False,
        "candidate_health_alerts": (
            ("log_accept_energy_proxy_exceeded",) if proxy_alert else ()
        ),
        "acceptance_decision_under_v3": (
            "unavailable_legacy_v2_early_return"
            if missing_acceptance
            else "unavailable_legacy_v2_requires_reanalysis"
        ),
        "repair_direction_under_v3": "unavailable",
        "source_payload_mutated": False,
    }


def evaluate_hmc_acceptance_evidence(
    *,
    samples: Any,
    log_accept_ratio: Any,
    is_accepted: Any,
    policy: HMCAcceptancePolicy,
    target_log_prob: Any | None = None,
    native_divergence_status: str = "not_exposed_by_kernel",
    native_divergence_count: int | None = None,
    candidate_local_health_failures: tuple[str, ...] = (),
    shared_invalidity_reasons: tuple[str, ...] = (),
    cost_stop_reasons: tuple[str, ...] = (),
) -> HMCAcceptanceEvidence:
    """Evaluate a checkpoint while keeping validity and tuning roles separate.

    ``candidate_local_health_failures`` is a legacy caller parameter. Its
    values are classified as candidate-data or shared invalidity by provenance;
    they never bypass the v3 role model or supply a repair direction.
    """

    import tensorflow as tf

    if not isinstance(policy, HMCAcceptancePolicy):
        raise TypeError("policy must be HMCAcceptancePolicy")
    sample_array = _float64_tensor(samples)
    log_accept = _float64_tensor(log_accept_ratio)
    accepted = tf.convert_to_tensor(is_accepted)
    if accepted.dtype != tf.bool:
        raise TypeError("is_accepted must be boolean")
    target_value = None if target_log_prob is None else _float64_tensor(target_log_prob)
    if sample_array.shape.rank == 2:
        sample_array = sample_array[:, None, :]
    if log_accept.shape.rank == 1:
        log_accept = log_accept[:, None]
    if accepted.shape.rank == 1:
        accepted = accepted[:, None]
    if target_value is not None and target_value.shape.rank == 1:
        target_value = target_value[:, None]
    if (
        sample_array.shape.rank != 3
        or log_accept.shape.rank != 2
        or accepted.shape.rank != 2
        or sample_array.shape[:2] != log_accept.shape
        or log_accept.shape != accepted.shape
        or (target_value is not None and target_value.shape != log_accept.shape)
    ):
        raise ValueError("samples and acceptance traces must have aligned draw/chain shapes")
    draw_count, chain_count = log_accept.shape
    status, divergence_count, divergence_provenance_valid = (
        _normalize_native_divergence_provenance(
            native_divergence_status,
            native_divergence_count,
        )
    )

    declared_shared = _sanitize_reason_codes(
        shared_invalidity_reasons,
        allowed=_SHARED_INVALIDITY_REASON_CODES,
    )
    if draw_count == 0 or chain_count == 0 or sample_array.shape[2] == 0:
        declared_shared += ("shared_schema_invalid",)
    legacy_failures = tuple(
        dict.fromkeys(str(item) for item in candidate_local_health_failures)
    )
    # This input is explicitly candidate-local. Shared scope is admitted only
    # through shared_invalidity_reasons or state-dependency checks below.
    shared_from_provenance: tuple[str, ...] = ()
    local_from_provenance = _sanitize_reason_codes(
        (
            item
            for item in legacy_failures
            if item not in _LEGACY_ROLE_ONLY_REASON_CODES
        ),
        allowed=_CANDIDATE_DATA_INVALIDITY_REASON_CODES,
    )
    if not _all_finite(sample_array):
        shared_from_provenance += ("nonfinite_retained_samples",)
    candidate_invalidity = list(local_from_provenance)
    if not divergence_provenance_valid:
        candidate_invalidity.append("native_divergence_provenance_inconsistent")
    if not _all_finite(log_accept):
        candidate_invalidity.append("nonfinite_log_accept_ratio")
    if target_value is not None and not _all_finite(target_value):
        candidate_invalidity.append("nonfinite_target_log_prob")
    shared = tuple(dict.fromkeys((*declared_shared, *shared_from_provenance)))
    if shared:
        return _empty_acceptance_evidence(
            evidence_validity="shared_execution_invalid",
            native_divergence_status=status,
            native_divergence_count=divergence_count,
            policy=policy,
            engineering_invalidity_reasons=shared,
        )
    if candidate_invalidity:
        return _empty_acceptance_evidence(
            evidence_validity="candidate_data_invalid",
            native_divergence_status=status,
            native_divergence_count=divergence_count,
            policy=policy,
            engineering_invalidity_reasons=tuple(dict.fromkeys(candidate_invalidity)),
        )

    summary = _acceptance_summary_function()(
        sample_array, log_accept, accepted,
        tf.constant(policy.block_count, tf.int32),
        tf.constant(policy.min_decisions_per_chain, tf.int32),
        tf.constant(policy.chain_count, tf.int32),
        tf.constant(policy.max_abs_log_accept_energy_proxy, tf.float64),
    )
    realized_acceptance_rate_by_chain = summary["realized_by_chain"]
    realized_acceptance_rate = float(summary["realized"])
    movement, repeated, normalized_return, path_return = summary["movement"]
    proxy = _signed_proxy_summary(log_accept, policy, summary=summary["proxy"])
    proxy_exceeded = (
        proxy["max_abs_log_accept_energy_proxy"]
        > policy.max_abs_log_accept_energy_proxy
    )
    alerts = (("log_accept_energy_proxy_exceeded",) if proxy_exceeded else ())
    followups = (
        ("inspect_exact_hamiltonian_and_signed_log_accept_tails",)
        if alerts
        else ()
    )
    promotion_vetoes = (
        ("native_divergence_positive",)
        if status == "available" and divergence_count is not None and divergence_count > 0
        else ()
    )
    movement_failed, path_return_failed = _trajectory_pathology_flags(
        policy=policy,
        movement=movement,
        repeated=repeated,
        normalized_return=normalized_return,
        path_return=path_return,
    )
    if movement_failed:
        promotion_vetoes = tuple(
            dict.fromkeys((*promotion_vetoes, "movement_gate_failed"))
        )
    if path_return_failed:
        promotion_vetoes = tuple(
            dict.fromkeys((*promotion_vetoes, "path_return_resonance_detected"))
        )
    if "native_divergence_positive" in legacy_failures and not promotion_vetoes:
        return _empty_acceptance_evidence(
            evidence_validity="candidate_data_invalid",
            native_divergence_status=status,
            native_divergence_count=divergence_count,
            policy=policy,
            engineering_invalidity_reasons=(
                "native_divergence_provenance_inconsistent",
            ),
        )
    if "log_accept_energy_proxy_exceeded" in legacy_failures and not proxy_exceeded:
        return _empty_acceptance_evidence(
            evidence_validity="candidate_data_invalid",
            native_divergence_status=status,
            native_divergence_count=divergence_count,
            policy=policy,
            engineering_invalidity_reasons=(
                "log_accept_proxy_provenance_inconsistent",
            ),
        )
    requested_cost_stops = tuple(dict.fromkeys(str(item) for item in cost_stop_reasons))
    unauthorized_cost_stops = set(requested_cost_stops).difference(
        policy.allowed_cost_stop_reasons
    )
    if unauthorized_cost_stops:
        raise ValueError("cost stop reason is not predeclared by the acceptance policy")
    if requested_cost_stops and (
        chain_count != policy.chain_count
        or draw_count < policy.min_decisions_per_chain
    ):
        raise ValueError("cost stop requires the minimum acceptance evidence")
    if chain_count != policy.chain_count or draw_count < policy.min_decisions_per_chain:
        return HMCAcceptanceEvidence(
            evidence_validity="valid",
            acceptance_decision="inconclusive_evidence",
            pooled_mean=float(summary["pooled"]),
            chain_mean_uncertainty_interval=None,
            chain_mean_uncertainty_method=None,
            chain_mean_uncertainty_level=None,
            chain_means=tuple(float(item) for item in summary["chain_means"]),
            block_means_by_chain=(),
            realized_acceptance_rate=realized_acceptance_rate,
            realized_acceptance_rate_by_chain=tuple(
                float(item) for item in realized_acceptance_rate_by_chain
            ),
            movement_rate_by_chain=tuple(float(item) for item in movement),
            repeated_state_fraction_by_chain=tuple(float(item) for item in repeated),
            normalized_return_displacement_by_chain=tuple(
                float(item) for item in normalized_return
            ),
            path_return_fraction_by_chain=tuple(
                float(item) for item in path_return
            ),
            usable_decisions_per_chain=0,
            excluded_remainder_per_chain=draw_count,
            native_divergence_status=status,
            native_divergence_count=divergence_count,
            **proxy,
            policy=policy,
            candidate_promotion_vetoes=promotion_vetoes,
            candidate_health_alerts=alerts,
            diagnostic_followups=followups,
            cost_stop_reasons=requested_cost_stops,
            explanatory_notes=("minimum_chain_or_decision_evidence_missing",),
        )

    usable = (draw_count // policy.block_count) * policy.block_count
    block_size = usable // policy.block_count
    if block_size < policy.min_block_size:
        raise AssertionError("minimum decision gate failed to imply minimum block size")
    block_means = summary["block_means"]
    chain_means = summary["chain_means"]
    pooled = float(summary["pooled"])
    interval = _chain_mean_uncertainty_interval(
        chain_means,
        policy=policy,
    )
    decision = _acceptance_decision_from_summary(
        policy=policy,
        pooled_mean=pooled,
        chain_means=chain_means,
        block_means=block_means,
        uncertainty_interval=interval,
        movement=movement,
        repeated=repeated,
        normalized_return=normalized_return,
        path_return=path_return,
    )
    repair_triggers = _repair_triggers_from_decision(
        decision,
        resonance_failed=path_return_failed,
    )
    return HMCAcceptanceEvidence(
        evidence_validity="valid",
        acceptance_decision=decision,
        pooled_mean=pooled,
        chain_mean_uncertainty_interval=interval,
        chain_mean_uncertainty_method=_CHAIN_MEAN_UNCERTAINTY_METHOD,
        chain_mean_uncertainty_level=policy.confidence_level,
        chain_means=tuple(float(item) for item in chain_means),
        block_means_by_chain=tuple(
            tuple(float(item) for item in row) for row in block_means
        ),
        realized_acceptance_rate=realized_acceptance_rate,
        realized_acceptance_rate_by_chain=tuple(
            float(item) for item in realized_acceptance_rate_by_chain
        ),
        movement_rate_by_chain=tuple(float(item) for item in movement),
        repeated_state_fraction_by_chain=tuple(float(item) for item in repeated),
        normalized_return_displacement_by_chain=tuple(
            float(item) for item in normalized_return
        ),
        path_return_fraction_by_chain=tuple(float(item) for item in path_return),
        usable_decisions_per_chain=usable,
        excluded_remainder_per_chain=draw_count - usable,
        native_divergence_status=status,
        native_divergence_count=divergence_count,
        **proxy,
        policy=policy,
        candidate_promotion_vetoes=promotion_vetoes,
        tuning_repair_triggers=repair_triggers,
        candidate_health_alerts=alerts,
        diagnostic_followups=followups,
        cost_stop_reasons=requested_cost_stops,
        explanatory_notes=("binary_acceptance_is_explanatory_only",),
    )


def _movement_summaries(
    samples: Any,
) -> tuple[Any, Any, Any, Any]:
    import tensorflow as tf

    samples = _float64_tensor(samples)

    def unavailable() -> tuple[Any, Any, Any, Any]:
        missing = tf.zeros([0], tf.float64)
        return missing, missing, missing, missing

    def available() -> tuple[Any, Any, Any, Any]:
        displacement = samples[1:] - samples[:-1]
        scale_by_coordinate = tf.math.reduce_std(samples, axis=0)
        threshold = tf.maximum(
            tf.constant(1.0e-12, tf.float64), 1.0e-10 * scale_by_coordinate
        )
        movement_by_coordinate = tf.reduce_mean(
            tf.cast(tf.abs(displacement) > threshold[None, :, :], tf.float64), axis=0
        )
        movement = tf.reduce_min(movement_by_coordinate, axis=-1)
        repeated = 1.0 - movement
        scale = tf.linalg.norm(scale_by_coordinate, axis=-1)
        normalized_return = tf.linalg.norm(samples[-1] - samples[0], axis=-1) / tf.maximum(
            scale, 1.0e-12
        )
        return movement, repeated, normalized_return, _path_return_fraction_by_chain(samples)

    return tf.cond(tf.shape(samples)[0] >= 2, available, unavailable)


def _path_return_fraction_by_chain(samples: Any) -> Any:
    """Maximum lag-2 through lag-16 recurrence, with one bounded graph body."""

    import tensorflow as tf

    samples = _float64_tensor(samples)
    draw_count = tf.shape(samples)[0]

    def available() -> Any:
        center = tf.reduce_mean(samples, axis=0, keepdims=True)

        def lag_fraction(lag: Any, maximum: Any) -> tuple[Any, Any]:
            current = samples[lag:]
            lagged = samples[:-lag]
            distance = tf.linalg.norm(current - lagged, axis=-1)
            state_scale = tf.maximum(
                tf.maximum(
                    tf.linalg.norm(current - center, axis=-1),
                    tf.linalg.norm(lagged - center, axis=-1),
                ),
                1.0,
            )
            threshold = _PATH_RETURN_ATOL + _PATH_RETURN_RTOL * state_scale
            fraction = tf.reduce_mean(tf.cast(distance <= threshold, tf.float64), axis=0)
            return lag + 1, tf.maximum(maximum, fraction)

        _, maximum = tf.while_loop(
            lambda lag, maximum: (lag <= max(_PATH_RETURN_LAGS)) & (lag < draw_count),
            lag_fraction,
            (tf.constant(min(_PATH_RETURN_LAGS)), tf.zeros([tf.shape(samples)[1]], tf.float64)),
        )
        return maximum

    return tf.cond(draw_count > min(_PATH_RETURN_LAGS), available, lambda: tf.zeros([0], tf.float64))


@lru_cache(maxsize=1)
def _acceptance_summary_function() -> Any:
    """Cache the CPU/non-XLA checkpoint arithmetic, polymorphic only in sizes.

    alpha=exp(min(log_accept_ratio,0)); four equal temporal blocks summarize
    each independent chain. The host wrapper owns role classification and
    serialization, and rejects empty/nonfinite inputs before calling this graph.
    These discarded tuning summaries do not measure posterior convergence.
    """

    import tensorflow as tf

    @tf.function(input_signature=(
        tf.TensorSpec([None, None, None], tf.float64),
        tf.TensorSpec([None, None], tf.float64),
        tf.TensorSpec([None, None], tf.bool),
        tf.TensorSpec([], tf.int32),
        tf.TensorSpec([], tf.int32),
        tf.TensorSpec([], tf.int32),
        tf.TensorSpec([], tf.float64),
    ), autograph=False, jit_compile=False)
    def summarize(samples, log_accept, accepted, blocks, minimum, chains, proxy_limit):
        probability = tf.exp(tf.minimum(log_accept, 0.0))
        draw_count = tf.shape(log_accept)[0]

        def complete():
            usable = draw_count // blocks * blocks
            block_means = tf.transpose(tf.reduce_mean(
                tf.reshape(probability[:usable], [blocks, usable // blocks, chains]), axis=1
            ))
            return tf.reduce_mean(block_means, axis=1), block_means

        means, block_means = tf.cond(
            (draw_count >= minimum) & (tf.shape(log_accept)[1] == chains),
            complete,
            lambda: (tf.reduce_mean(probability, axis=0), tf.zeros([0, 0], tf.float64)),
        )
        realized_by_chain = tf.reduce_mean(tf.cast(accepted, tf.float64), axis=0)
        return {
            "chain_means": means,
            "pooled": tf.reduce_mean(means),
            "block_means": block_means,
            "realized_by_chain": realized_by_chain,
            "realized": tf.reduce_mean(realized_by_chain),
            "movement": _movement_summaries(samples),
            "proxy": _signed_proxy_tensors(log_accept, proxy_limit),
        }

    return summarize


def _chain_mean_uncertainty_interval(
    chain_means: Any,
    *,
    policy: HMCAcceptancePolicy,
) -> tuple[float, float]:
    """Return the v5 working interval over independent chain means.

    The four seeded chains, rather than individual transitions or temporal
    blocks, are the declared independent units.  This is a bounded tuning
    compatibility interval; it is not a convergence or posterior interval.
    """

    import tensorflow as tf

    values = _float64_tensor(chain_means)
    if values.shape.rank != 1 or values.shape[0] != policy.chain_count:
        raise ValueError("chain-mean interval requires exactly four chain means")
    if not _all_finite(values):
        raise ValueError("chain means must be finite for uncertainty calculation")
    pooled_tensor = tf.reduce_mean(values)
    pooled = float(pooled_tensor)
    sample_sd = tf.sqrt(tf.reduce_sum(tf.square(values - pooled_tensor)) / (policy.chain_count - 1))
    half_width = (
        policy.student_critical_value
        * sample_sd
        / tf.sqrt(tf.cast(policy.chain_count, tf.float64))
    )
    return (
        max(0.0, pooled - float(half_width)),
        min(1.0, pooled + float(half_width)),
    )


def _empty_acceptance_evidence(
    *,
    evidence_validity: str,
    native_divergence_status: str,
    native_divergence_count: int | None,
    policy: HMCAcceptancePolicy,
    engineering_invalidity_reasons: tuple[str, ...],
) -> HMCAcceptanceEvidence:
    return HMCAcceptanceEvidence(
        evidence_validity=evidence_validity,
        acceptance_decision="unavailable",
        pooled_mean=None,
        chain_mean_uncertainty_interval=None,
        chain_mean_uncertainty_method=None,
        chain_mean_uncertainty_level=None,
        chain_means=(),
        block_means_by_chain=(),
        realized_acceptance_rate=None,
        realized_acceptance_rate_by_chain=(),
        movement_rate_by_chain=(),
        repeated_state_fraction_by_chain=(),
        normalized_return_displacement_by_chain=(),
        path_return_fraction_by_chain=(),
        usable_decisions_per_chain=0,
        excluded_remainder_per_chain=0,
        native_divergence_status=native_divergence_status,
        native_divergence_count=native_divergence_count,
        min_log_accept_ratio=None,
        max_log_accept_ratio=None,
        max_abs_log_accept_energy_proxy=None,
        negative_proxy_exceedance_count_by_chain=(),
        positive_proxy_exceedance_count_by_chain=(),
        negative_proxy_exceedance_rate_by_chain=(),
        positive_proxy_exceedance_rate_by_chain=(),
        policy=policy,
        engineering_invalidity_reasons=engineering_invalidity_reasons,
    )


def _signed_proxy_summary(
    log_accept_ratio: Any,
    policy: HMCAcceptancePolicy,
    *,
    summary: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if summary is None:
        summary = _signed_proxy_tensors(
            _float64_tensor(log_accept_ratio), policy.max_abs_log_accept_energy_proxy
        )
    negative_counts = summary["negative"]
    positive_counts = summary["positive"]
    draw_count = int(log_accept_ratio.shape[0])
    denominator = float(draw_count)
    return {
        "min_log_accept_ratio": float(summary["minimum"]),
        "max_log_accept_ratio": float(summary["maximum"]),
        "max_abs_log_accept_energy_proxy": float(summary["max_abs"]),
        "negative_proxy_exceedance_count_by_chain": tuple(
            int(item) for item in negative_counts
        ),
        "positive_proxy_exceedance_count_by_chain": tuple(
            int(item) for item in positive_counts
        ),
        "negative_proxy_exceedance_rate_by_chain": tuple(
            float(item) / denominator for item in negative_counts
        ),
        "positive_proxy_exceedance_rate_by_chain": tuple(
            float(item) / denominator for item in positive_counts
        ),
    }


def _signed_proxy_tensors(log_accept_ratio: Any, threshold: Any) -> Mapping[str, Any]:
    import tensorflow as tf

    return {
        "minimum": tf.reduce_min(log_accept_ratio),
        "maximum": tf.reduce_max(log_accept_ratio),
        "max_abs": tf.reduce_max(tf.abs(log_accept_ratio)),
        "negative": tf.reduce_sum(tf.cast(log_accept_ratio < -threshold, tf.int64), axis=0),
        "positive": tf.reduce_sum(tf.cast(log_accept_ratio > threshold, tf.int64), axis=0),
    }


def _normalize_native_divergence_provenance(
    status: Any,
    count: Any,
) -> tuple[str, int | None, bool]:
    normalized_status = str(status)
    allowed_statuses = {"available", "not_exposed_by_kernel", "not_collected"}
    count_is_integer = isinstance(count, Integral) and not isinstance(count, bool)
    normalized_count = int(count) if count_is_integer else None
    valid = (
        normalized_status in allowed_statuses
        and (
            (normalized_status == "available" and normalized_count is not None)
            or (normalized_status != "available" and count is None)
        )
        and (normalized_count is None or normalized_count >= 0)
    )
    if not valid:
        return "not_collected", None, False
    return normalized_status, normalized_count, True


def _sanitize_reason_codes(
    reasons: Any,
    *,
    allowed: frozenset[str],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            code if code in allowed else "unrecognized_health_failure"
            for code in (str(item) for item in reasons)
        )
    )


def _repair_triggers_from_decision(
    decision: str,
    *,
    resonance_failed: bool = False,
) -> tuple[str, ...]:
    trigger = {
        "repair_step_lower": "step_size:lower_epsilon",
        "repair_step_higher": "step_size:higher_epsilon",
        "repair_trajectory": "trajectory:repair_movement",
    }.get(str(decision))
    if str(decision) == "repair_trajectory" and resonance_failed:
        trigger = "trajectory:repair_resonance"
    return () if trigger is None else (trigger,)


def _validate_signed_proxy_summary(evidence: HMCAcceptanceEvidence) -> None:
    extrema = (
        evidence.min_log_accept_ratio,
        evidence.max_log_accept_ratio,
        evidence.max_abs_log_accept_energy_proxy,
    )
    has_extrema = all(item is not None for item in extrema)
    has_any_extrema = any(item is not None for item in extrema)
    count_names = (
        "negative_proxy_exceedance_count_by_chain",
        "positive_proxy_exceedance_count_by_chain",
    )
    rate_names = (
        "negative_proxy_exceedance_rate_by_chain",
        "positive_proxy_exceedance_rate_by_chain",
    )
    counts = []
    for name in count_names:
        values = tuple(
            _strict_scalar_integer(item, name=f"{name} item")
            for item in getattr(evidence, name)
        )
        if any(item < 0 for item in values):
            raise ValueError(f"{name} must be nonnegative")
        object.__setattr__(evidence, name, values)
        counts.append(values)
    rates = []
    for name in rate_names:
        values = tuple(float(item) for item in getattr(evidence, name))
        if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in values):
            raise ValueError(f"{name} must lie inside [0, 1]")
        object.__setattr__(evidence, name, values)
        rates.append(values)
    if has_any_extrema != has_extrema:
        raise ValueError("signed proxy extrema must be jointly available")
    if has_extrema:
        minimum, maximum, max_abs = (float(item) for item in extrema)
        if not all(math.isfinite(item) for item in (minimum, maximum, max_abs)):
            raise ValueError("signed proxy extrema must be finite")
        if minimum > maximum or max_abs < 0.0:
            raise ValueError("signed proxy extrema are inconsistent")
        if not _all_close(max_abs, max(abs(minimum), abs(maximum)), rtol=1e-12, atol=1e-12):
            raise ValueError("absolute proxy maximum does not match signed extrema")
        object.__setattr__(evidence, "min_log_accept_ratio", minimum)
        object.__setattr__(evidence, "max_log_accept_ratio", maximum)
        object.__setattr__(evidence, "max_abs_log_accept_energy_proxy", max_abs)
        chain_count = len(evidence.chain_means)
        if chain_count == 0 or any(len(items) != chain_count for items in (*counts, *rates)):
            raise ValueError("signed proxy summaries must align with acceptance chains")
        draw_count = evidence.usable_decisions_per_chain + evidence.excluded_remainder_per_chain
        if draw_count <= 0:
            raise ValueError("signed proxy summaries require a positive draw count")
        for count_values, rate_values in zip(counts, rates):
            expected_rates = _float64_tensor(count_values) / float(draw_count)
            if not _all_close(rate_values, expected_rates, rtol=1e-12, atol=1e-12):
                raise ValueError("signed proxy rates do not match their counts")
    elif any((*counts, *rates)):
        if any(values for values in (*counts, *rates)):
            raise ValueError("signed proxy counts require extrema")


def _validate_valid_acceptance_summary(
    evidence: HMCAcceptanceEvidence,
    block_means: tuple[tuple[float, ...], ...],
) -> None:
    import tensorflow as tf

    decision = evidence.acceptance_decision
    draw_count = (
        evidence.usable_decisions_per_chain
        + evidence.excluded_remainder_per_chain
    )
    realized_rates = _float64_tensor(evidence.realized_acceptance_rate_by_chain)
    realized_counts = realized_rates * draw_count
    if (
        evidence.realized_acceptance_rate is None
        or len(evidence.realized_acceptance_rate_by_chain) != len(evidence.chain_means)
        or draw_count <= 0
        or not _all_close(realized_counts, tf.round(realized_counts), rtol=0.0, atol=1e-12)
        or not _all_close(
            evidence.realized_acceptance_rate,
            tf.reduce_mean(realized_rates),
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise ValueError("realized acceptance summary is incomplete or inconsistent")
    has_interval = evidence.chain_mean_uncertainty_interval is not None
    if not has_interval:
        if decision != "inconclusive_evidence":
            raise ValueError("acceptance decision requires complete evidence")
        if evidence.pooled_mean is None or not evidence.chain_means:
            raise ValueError("valid incomplete evidence requires descriptive means")
        if block_means:
            raise ValueError("incomplete evidence cannot carry block components")
        return
    if (
        evidence.pooled_mean is None
        or len(evidence.chain_means) != evidence.policy.chain_count
        or len(block_means) != evidence.policy.chain_count
        or any(len(row) != evidence.policy.block_count for row in block_means)
        or len(evidence.movement_rate_by_chain) != evidence.policy.chain_count
        or len(evidence.repeated_state_fraction_by_chain) != evidence.policy.chain_count
        or len(evidence.normalized_return_displacement_by_chain) != evidence.policy.chain_count
        or len(evidence.path_return_fraction_by_chain) != evidence.policy.chain_count
        or evidence.usable_decisions_per_chain < evidence.policy.min_decisions_per_chain
        or evidence.usable_decisions_per_chain % evidence.policy.block_count != 0
        or evidence.excluded_remainder_per_chain >= evidence.policy.block_count
    ):
        raise ValueError("acceptance decision requires four-chain four-block evidence")
    expected_chain_means = tf.reduce_mean(_float64_tensor(block_means), axis=1)
    expected_pooled = float(tf.reduce_mean(expected_chain_means))
    expected_interval = _chain_mean_uncertainty_interval(
        expected_chain_means,
        policy=evidence.policy,
    )
    if not _all_close(evidence.chain_means, expected_chain_means, rtol=1e-12, atol=1e-12):
        raise ValueError("chain_means do not match block means")
    if not _all_close(evidence.pooled_mean, expected_pooled, rtol=1e-12, atol=1e-12):
        raise ValueError("pooled_mean does not match chain means")
    if not _all_close(
        evidence.chain_mean_uncertainty_interval,
        expected_interval,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("chain-mean uncertainty interval is inconsistent")
    if (
        evidence.chain_mean_uncertainty_method != _CHAIN_MEAN_UNCERTAINTY_METHOD
        or evidence.chain_mean_uncertainty_level != evidence.policy.confidence_level
    ):
        raise ValueError("chain-mean uncertainty metadata is inconsistent")
    expected_decision = _acceptance_decision_from_summary(
        policy=evidence.policy,
        pooled_mean=expected_pooled,
        chain_means=expected_chain_means,
        block_means=_float64_tensor(block_means),
        uncertainty_interval=expected_interval,
        movement=_float64_tensor(evidence.movement_rate_by_chain),
        repeated=_float64_tensor(evidence.repeated_state_fraction_by_chain),
        normalized_return=_float64_tensor(evidence.normalized_return_displacement_by_chain),
        path_return=_float64_tensor(evidence.path_return_fraction_by_chain),
    )
    if decision != expected_decision:
        raise ValueError("decision is inconsistent with acceptance policy")


def _validate_v5_roles(evidence: HMCAcceptanceEvidence) -> None:
    expected_vetoes = []
    if (
        evidence.native_divergence_status == "available"
        and evidence.native_divergence_count is not None
        and evidence.native_divergence_count > 0
    ):
        expected_vetoes.append("native_divergence_positive")
    movement_failed, path_return_failed = _trajectory_pathology_flags(
        policy=evidence.policy,
        movement=_float64_tensor(evidence.movement_rate_by_chain),
        repeated=_float64_tensor(evidence.repeated_state_fraction_by_chain),
        normalized_return=_float64_tensor(
            evidence.normalized_return_displacement_by_chain
        ),
        path_return=_float64_tensor(evidence.path_return_fraction_by_chain),
    )
    expected_triggers = _repair_triggers_from_decision(
        evidence.acceptance_decision,
        resonance_failed=path_return_failed,
    )
    if evidence.tuning_repair_triggers != expected_triggers:
        raise ValueError("tuning repair triggers are inconsistent with acceptance decision")
    if movement_failed:
        expected_vetoes.append("movement_gate_failed")
    if path_return_failed:
        expected_vetoes.append("path_return_resonance_detected")
    if evidence.candidate_promotion_vetoes != tuple(expected_vetoes):
        raise ValueError("candidate promotion vetoes are inconsistent with evidence")
    proxy_exceeded = (
        evidence.max_abs_log_accept_energy_proxy is not None
        and evidence.max_abs_log_accept_energy_proxy
        > evidence.policy.max_abs_log_accept_energy_proxy
    )
    expected_alerts = (("log_accept_energy_proxy_exceeded",) if proxy_exceeded else ())
    expected_followups = (
        ("inspect_exact_hamiltonian_and_signed_log_accept_tails",)
        if proxy_exceeded
        else ()
    )
    if evidence.candidate_health_alerts != expected_alerts:
        raise ValueError("candidate health alerts are inconsistent with telemetry")
    if evidence.diagnostic_followups != expected_followups:
        raise ValueError("diagnostic followups are inconsistent with telemetry")
    if not set(evidence.cost_stop_reasons).issubset(evidence.policy.allowed_cost_stop_reasons):
        raise ValueError("cost stop reasons were not predeclared by policy")


def _validate_hmc_acceptance_evidence_v2_payload(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("legacy acceptance evidence payload must be a mapping")
    if payload.get("schema") != "bayesfilter.hmc_acceptance_evidence.v2":
        raise ValueError("legacy acceptance evidence schema mismatch")
    decision = str(payload.get("decision", ""))
    if decision not in {
        "passed",
        "repair_step_lower",
        "repair_step_higher",
        "repair_trajectory",
        "inconclusive_conflict",
        "inconclusive_evidence",
        "candidate_local_veto",
        "shared_invalidity",
    }:
        raise ValueError("legacy acceptance decision is unsupported")
    policy = payload.get("policy")
    if not isinstance(policy, Mapping) or policy.get("schema") != "bayesfilter.hmc_acceptance_policy.v2":
        raise ValueError("legacy acceptance policy schema mismatch")
    if payload.get("passed") is not (decision == "passed"):
        raise ValueError("legacy acceptance passed flag is inconsistent")
    expected_direction = {
        "repair_step_lower": "lower_epsilon",
        "repair_step_higher": "higher_epsilon",
    }.get(decision)
    if payload.get("repair_direction") != expected_direction:
        raise ValueError("legacy acceptance repair direction is inconsistent")
    if payload.get("raw_traces_exposed") is not False:
        raise ValueError("legacy acceptance evidence unexpectedly exposes raw traces")
    if payload.get("reports_posterior_convergence") is not False:
        raise ValueError("legacy acceptance evidence claims posterior convergence")
    return payload


def summarize_hmc_tuning_telemetry(
    *,
    samples: Any,
    log_accept_ratio: Any,
    is_accepted: Any,
) -> Mapping[str, Any]:
    """Return per-chain acceptance-probability, movement, and energy telemetry.

    Inputs use leading draw dimension. A rank-2 sample tensor is a single chain
    with trailing parameter dimension; rank 3 is ``[draw, chain, parameter]``.
    Log acceptance and accept/reject tensors are correspondingly rank 1 or 2.
    """

    import tensorflow as tf

    sample_tensor = _float64_tensor(samples)
    log_accept = _float64_tensor(log_accept_ratio)
    accepted = tf.convert_to_tensor(is_accepted)
    if accepted.dtype != tf.bool:
        raise TypeError("is_accepted must be boolean")
    if sample_tensor.shape.rank == 2:
        sample_tensor = sample_tensor[:, tf.newaxis, :]
    elif sample_tensor.shape.rank != 3:
        raise ValueError("samples must have rank 2 or 3")
    if log_accept.shape.rank == 1:
        log_accept = log_accept[:, tf.newaxis]
    elif log_accept.shape.rank != 2:
        raise ValueError("log_accept_ratio must have rank 1 or 2")
    if accepted.shape.rank == 1:
        accepted = accepted[:, tf.newaxis]
    elif accepted.shape.rank != 2:
        raise ValueError("is_accepted must have rank 1 or 2")
    if sample_tensor.shape[0] != log_accept.shape[0] or log_accept.shape != accepted.shape:
        raise ValueError("telemetry draw and chain shapes must agree")
    if sample_tensor.shape[1] != log_accept.shape[1]:
        raise ValueError("sample and acceptance chain counts must agree")

    return {
        "schema": "bayesfilter.hmc_tuning_telemetry.v4",
        **_telemetry_summary_function()(sample_tensor, log_accept, accepted),
        "energy_proxy_role": "absolute_log_accept_ratio_not_hamiltonian_energy_error",
        "diagnostic_roles": {
            "mean_acceptance_probability": "promotion_criterion_and_repair_trigger",
            "binary_acceptance_rate": "explanatory_movement_diagnostic",
            "movement_rate_by_chain": "promotion_veto_and_repair_trigger",
            "path_return_fraction_by_chain": (
                "promotion_veto_and_resonance_repair_trigger"
            ),
            "max_abs_log_accept_energy_proxy": "explanatory_alert_only",
            "signed_log_accept_ratio_tails": "explanatory_alert_only",
        },
        "nonclaims": (
            "bounded kernel-tuning telemetry only",
            "absolute log acceptance is not native Hamiltonian energy error",
            "no posterior convergence claim",
        ),
    }


@lru_cache(maxsize=1)
def _telemetry_summary_function() -> Any:
    import tensorflow as tf

    return tf.function(
        _tuning_telemetry_tensors,
        input_signature=(
            tf.TensorSpec([None, None, None], tf.float64),
            tf.TensorSpec([None, None], tf.float64),
            tf.TensorSpec([None, None], tf.bool),
        ),
        autograph=False,
        jit_compile=False,
    )


def _tuning_telemetry_tensors(sample_tensor: Any, log_accept: Any, accepted: Any) -> Mapping[str, Any]:
    """Numeric post-run telemetry; nonfinite log ratios remain visible."""

    import tensorflow as tf

    finite_log_accept = tf.math.is_finite(log_accept)
    acceptance_probability = tf.exp(tf.minimum(log_accept, 0.0))
    acceptance_probability = tf.where(
        finite_log_accept,
        acceptance_probability,
        tf.fill(tf.shape(acceptance_probability), tf.constant(float("nan"), tf.float64)),
    )
    mean_acceptance_by_chain = tf.reduce_mean(acceptance_probability, axis=0)
    binary_acceptance_by_chain = tf.reduce_mean(tf.cast(accepted, tf.float64), axis=0)

    draw_count = tf.shape(sample_tensor, out_type=tf.int32)[0]

    def movement_with_pairs() -> tuple[Any, Any, Any, Any]:
        displacement = sample_tensor[1:] - sample_tensor[:-1]
        l2 = tf.linalg.norm(displacement, axis=-1)
        scale_by_coordinate = tf.math.reduce_std(sample_tensor, axis=0)
        movement_threshold = tf.maximum(
            tf.constant(1.0e-12, tf.float64),
            tf.constant(1.0e-10, tf.float64) * scale_by_coordinate,
        )
        movement_by_coordinate = tf.reduce_mean(
            tf.cast(
                tf.abs(displacement) > movement_threshold[tf.newaxis, :, :],
                tf.float64,
            ),
            axis=0,
        )
        movement_rate = tf.reduce_min(movement_by_coordinate, axis=-1)
        repeated_fraction = tf.constant(1.0, tf.float64) - movement_rate
        mean_l2 = tf.reduce_mean(l2, axis=0)
        scale_norm = tf.linalg.norm(scale_by_coordinate, axis=-1)
        return_displacement = tf.linalg.norm(sample_tensor[-1] - sample_tensor[0], axis=-1)
        normalized_return = return_displacement / tf.maximum(
            scale_norm,
            tf.constant(1.0e-12, tf.float64),
        )
        return movement_rate, repeated_fraction, mean_l2, normalized_return

    def movement_without_pairs() -> tuple[Any, Any, Any, Any]:
        missing = tf.zeros([0], tf.float64)
        return missing, missing, missing, missing

    movement_rate, repeated_fraction, mean_l2, normalized_return = tf.cond(
        draw_count > 1,
        movement_with_pairs,
        movement_without_pairs,
    )

    path_return_fraction = _path_return_fraction_by_chain(sample_tensor)
    finite_log_values = tf.boolean_mask(log_accept, finite_log_accept)
    finite_energy = tf.abs(finite_log_values)
    max_abs_log_accept = tf.cond(
        tf.size(finite_energy) > 0,
        lambda: tf.reduce_max(finite_energy),
        lambda: tf.constant(float("nan"), tf.float64),
    )
    min_log_accept = tf.cond(
        tf.size(finite_log_values) > 0,
        lambda: tf.reduce_min(finite_log_values),
        lambda: tf.constant(float("nan"), tf.float64),
    )
    max_log_accept = tf.cond(
        tf.size(finite_log_values) > 0,
        lambda: tf.reduce_max(finite_log_values),
        lambda: tf.constant(float("nan"), tf.float64),
    )
    proxy_threshold = tf.constant(1000.0, tf.float64)
    negative_proxy_counts = tf.reduce_sum(
        tf.cast(log_accept < -proxy_threshold, tf.int32), axis=0
    )
    positive_proxy_counts = tf.reduce_sum(
        tf.cast(log_accept > proxy_threshold, tf.int32), axis=0
    )
    draw_count_float = tf.cast(draw_count, tf.float64)
    return {
        "mean_acceptance_probability": tf.reduce_mean(acceptance_probability),
        "mean_acceptance_probability_by_chain": mean_acceptance_by_chain,
        "binary_acceptance_rate": tf.reduce_mean(tf.cast(accepted, tf.float64)),
        "binary_acceptance_rate_by_chain": binary_acceptance_by_chain,
        "movement_rate_by_chain": movement_rate,
        "repeated_state_fraction_by_chain": repeated_fraction,
        "mean_displacement_l2_by_chain": mean_l2,
        "normalized_return_displacement_by_chain": normalized_return,
        "path_return_fraction_by_chain": path_return_fraction,
        "log_accept_ratio_finite_count": tf.reduce_sum(
            tf.cast(finite_log_accept, tf.int32)
        ),
        "log_accept_ratio_nonfinite_count": tf.reduce_sum(
            tf.cast(tf.logical_not(finite_log_accept), tf.int32)
        ),
        "min_log_accept_ratio": min_log_accept,
        "max_log_accept_ratio": max_log_accept,
        "max_abs_log_accept_energy_proxy": max_abs_log_accept,
        "negative_proxy_exceedance_count_by_chain": negative_proxy_counts,
        "positive_proxy_exceedance_count_by_chain": positive_proxy_counts,
        "negative_proxy_exceedance_rate_by_chain": (
            tf.cast(negative_proxy_counts, tf.float64) / draw_count_float
        ),
        "positive_proxy_exceedance_rate_by_chain": (
            tf.cast(positive_proxy_counts, tf.float64) / draw_count_float
        ),
        "proxy_alert_threshold": proxy_threshold,
        "draw_count": draw_count,
        "chain_count": tf.shape(sample_tensor, out_type=tf.int32)[1],
        "movement_summary_available": draw_count > 1,
    }


def _acceptance_decision_from_summary(
    *,
    policy: HMCAcceptancePolicy,
    pooled_mean: float,
    chain_means: Any,
    block_means: Any,
    uncertainty_interval: tuple[float, float],
    movement: Any,
    repeated: Any,
    normalized_return: Any,
    path_return: Any,
) -> str:
    import tensorflow as tf

    chain_means = _float64_tensor(chain_means)
    block_means = _float64_tensor(block_means)
    low, high = policy.practical_region
    repair_low, repair_high = policy.repair_region
    interval_low, interval_high = (float(item) for item in uncertainty_interval)
    low_supported = interval_high < low
    high_supported = interval_low > high
    chain_conflict = bool(tf.reduce_any(chain_means < low) and tf.reduce_any(chain_means > high))
    temporal_block_conflict = bool(
        tf.reduce_any(
            (tf.reduce_min(block_means, axis=1) < low)
            & (tf.reduce_max(block_means, axis=1) > high)
        )
    )
    movement_failed, path_return_failed = _trajectory_pathology_flags(
        policy=policy,
        movement=movement,
        repeated=repeated,
        normalized_return=normalized_return,
        path_return=path_return,
    )
    trajectory_pathology = movement_failed or path_return_failed
    # Rejections caused by an oversized step also create repeated states. Honor
    # the supported scalar repair before interpreting that symptom as an
    # independent trajectory defect; the pathology remains a promotion veto.
    if low_supported:
        return "repair_step_lower"
    if chain_conflict:
        return "inconclusive_conflict"
    if temporal_block_conflict:
        return "inconclusive_evidence"
    if trajectory_pathology:
        return "repair_trajectory"
    if high_supported:
        return "repair_step_higher"
    if (
        interval_low <= high
        and interval_high >= low
        and interval_low >= repair_low
        and interval_high <= repair_high
        and bool(tf.reduce_all((chain_means >= repair_low) & (chain_means <= repair_high)))
    ):
        return "passed"
    return "inconclusive_evidence"


def _trajectory_pathology_flags(
    *,
    policy: HMCAcceptancePolicy,
    movement: Any,
    repeated: Any,
    normalized_return: Any,
    path_return: Any,
) -> tuple[bool, bool]:
    import tensorflow as tf

    movement = _float64_tensor(movement)
    repeated = _float64_tensor(repeated)
    normalized_return = _float64_tensor(normalized_return)
    path_return = _float64_tensor(path_return)
    movement_failed = bool(
        tf.reduce_any(movement < policy.min_movement_rate)
        or tf.reduce_any(repeated > policy.max_repeated_state_fraction)
        or tf.reduce_any(normalized_return < policy.min_normalized_return_displacement)
    )
    # A stuck chain is trivially equal to itself at every lag. Reserve the
    # resonance label for recurrent paths that make real adjacent-state moves.
    moving_chain = (
        (movement >= policy.min_movement_rate)
        & (repeated <= policy.max_repeated_state_fraction)
    )
    path_return_failed = bool(
        int(tf.size(path_return))
        and tf.reduce_any(
            (path_return > _PATH_RETURN_MAX_FRACTION)
            & moving_chain
        )
    )
    return movement_failed, path_return_failed


def _acceptance_policy_from_payload(payload: Any) -> HMCAcceptancePolicy:
    if not isinstance(payload, Mapping):
        raise TypeError("acceptance policy payload must be a mapping")
    if payload.get("schema") != "bayesfilter.hmc_acceptance_policy.v5":
        raise ValueError("acceptance policy schema mismatch")
    expected_keys = set(HMCAcceptancePolicy().payload())
    if set(payload) != expected_keys:
        raise ValueError("acceptance policy field set is inconsistent")
    policy = HMCAcceptancePolicy(
        target=payload.get("target"),
        practical_region=tuple(payload.get("practical_region", ())),
        repair_region=tuple(payload.get("repair_region", ())),
        chain_count=payload.get("chain_count"),
        block_count=payload.get("block_count"),
        min_block_size=payload.get("min_block_size"),
        confidence_level=payload.get("confidence_level"),
        min_movement_rate=payload.get("min_movement_rate"),
        max_repeated_state_fraction=payload.get("max_repeated_state_fraction"),
        min_normalized_return_displacement=payload.get(
            "min_normalized_return_displacement"
        ),
        max_abs_log_accept_energy_proxy=payload.get(
            "max_abs_log_accept_energy_proxy"
        ),
        allowed_cost_stop_reasons=tuple(
            payload.get("allowed_cost_stop_reasons", ())
        ),
    )
    expected = policy.payload()
    for name in expected:
        if _container_normalized(payload.get(name)) != _container_normalized(
            expected[name]
        ):
            raise ValueError(f"acceptance policy {name} is inconsistent")
    return policy


def _container_normalized(value: Any) -> Any:
    """Normalize JSON list/tuple representation without coercing scalar values."""

    if isinstance(value, Mapping):
        return tuple(
            (str(key), _container_normalized(item))
            for key, item in sorted(value.items())
        )
    if isinstance(value, (tuple, list)):
        return tuple(_container_normalized(item) for item in value)
    return value
