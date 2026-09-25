"""Private replay and hard-boundary artifacts for HMC tuning engineering runs.

This module contains no sampler or numerical policy. It validates sanitized
summaries emitted by the operational TF/TFP tuning route and lets a parent
process enforce a real wall-clock cap around a child command.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from bayesfilter.hmc_route_contract import (
    HMC_ROUTE_CONTRACT_VERSION,
    OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
)
from bayesfilter.inference.hmc_coordinates import WarmupTrajectoryPolicy
from bayesfilter.inference.hmc_tuning import (
    WindowedMassAdaptationConfig,
    build_windowed_warmup_schedule,
)
from bayesfilter.inference.hmc_tuning_state import HMCTuningTransition
from bayesfilter.inference.hmc_verification import (
    hmc_acceptance_evidence_from_payload,
    hmc_acceptance_evidence_v2_migration_view,
)
from bayesfilter.inference.hmc_warmup import OPERATIONAL_WARMUP_NONCLAIMS


TUNING_ARTIFACT_NONCLAIMS = (
    "bounded HMC tuning engineering artifact only",
    "no posterior convergence claim",
    "no sampler superiority claim",
    "no default-readiness claim",
    "no GPU or XLA readiness claim",
    "no scientific claim",
)

_ARTIFACT_SCHEMA = "bayesfilter.hmc_tuning_engineering_artifact.v3"
_LEGACY_ARTIFACT_SCHEMA = "bayesfilter.hmc_tuning_engineering_artifact.v2"
_TRANSITION_SCHEMA = "bayesfilter.hmc_tuning_transition_ledger.v2"
_KERNEL_SCHEMA = "bayesfilter.hmc_kernel_state.v2"
_START_BANK_SCHEMA = "bayesfilter.hmc_private_start_bank.v2"
_TRAJECTORY_HANDOFF_SCHEMA = "bayesfilter.hmc_trajectory_handoff.v2"
_TIMEOUT_SCHEMA = "bayesfilter.hmc_killable_child_closeout.v2"


def _strict_scalar_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer scalar")
    return int(value)


def _strict_seed(value: Any, *, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two integer scalars")
    return tuple(
        _strict_scalar_integer(item, name=f"{name} item") for item in value
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(encoded, encoding="ascii")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def kernel_state_summary(kernel_state: Any) -> Mapping[str, Any]:
    transform = kernel_state.transform
    metric = kernel_state.momentum_metric
    trajectory = kernel_state.trajectory_policy
    epsilon = kernel_state.epsilon
    if epsilon is None:
        raise ValueError("frozen kernel summary requires epsilon")
    return {
        "schema": _KERNEL_SCHEMA,
        "coordinate_signature": transform.signature,
        "metric_signature": metric.signature,
        "trajectory_signature": trajectory.signature,
        "epsilon_context_signature": kernel_state.epsilon_context_signature,
        "epsilon": float(epsilon),
        "adaptation_generation": int(kernel_state.adaptation_generation),
        "seed_lineage": tuple(int(item) for item in kernel_state.seed_lineage),
        "evidence_status": str(kernel_state.evidence_status),
        "canonical_theta_signature": canonical_sha256(
            {
                "schema": "bayesfilter.hmc_private_canonical_theta.v2",
                "values": kernel_state.canonical_theta.tolist(),
            }
        ),
        "raw_state_values_exposed": False,
    }


def private_start_bank_summary(
    operational_warmup: Any,
    *,
    active_signature: str | None = None,
) -> Mapping[str, Any]:
    source_signature = str(operational_warmup.private_start_bank_signature)
    signature = source_signature if active_signature is None else str(active_signature)
    if not source_signature or not signature:
        raise ValueError("private start-bank signatures must be non-empty")
    return {
        "schema": _START_BANK_SCHEMA,
        "signature": signature,
        "source_signature": source_signature,
        "count": 4,
        "coordinate_signature": operational_warmup.final_kernel_state.transform.signature,
        "seed_root": tuple(int(item) for item in operational_warmup.seed_root),
        "raw_values_exposed": False,
        "paths_exposed": False,
    }


def _finite_real(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _validate_reasonable_epsilon_payload(
    payload: Any,
    *,
    name: str,
) -> float:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} is missing")
    base_fields = {
        "status",
        "selected_step_size",
        "attempts",
        "diagnostic_role",
        "nonclaims",
    }
    status = payload.get("status")
    if status == "externally_qualified":
        expected_fields = base_fields | {"qualification_source"}
        if set(payload) != expected_fields:
            raise ValueError(f"{name} field set or status is invalid")
        source = payload.get("qualification_source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{name} qualification source is missing")
        attempts = payload.get("attempts")
        if not isinstance(attempts, (tuple, list)) or attempts:
            raise ValueError(f"{name} externally qualified attempts are invalid")
        _finite_real(
            payload.get("selected_step_size"),
            name=f"{name} selected_step_size",
            positive=True,
        )
        if payload.get("diagnostic_role") != "reasonable_epsilon_engineering_bracket":
            raise ValueError(f"{name} diagnostic role is invalid")
        if tuple(str(item) for item in payload.get("nonclaims", ())) != (
            OPERATIONAL_WARMUP_NONCLAIMS
        ):
            raise ValueError(f"{name} nonclaims changed")
        return float(payload["selected_step_size"])
    if set(payload) != base_fields or status != "passed":
        raise ValueError(f"{name} field set or status is invalid")
    if payload.get("diagnostic_role") != "reasonable_epsilon_engineering_bracket":
        raise ValueError(f"{name} diagnostic role is invalid")
    if tuple(str(item) for item in payload.get("nonclaims", ())) != (
        OPERATIONAL_WARMUP_NONCLAIMS
    ):
        raise ValueError(f"{name} nonclaims changed")
    selected = _finite_real(
        payload.get("selected_step_size"),
        name=f"{name} selected_step_size",
        positive=True,
    )
    attempts = payload.get("attempts")
    if not isinstance(attempts, (tuple, list)) or not attempts:
        raise ValueError(f"{name} attempts are missing")
    allowed_health = {"target_status_telemetry_failure"}
    final_step = None
    final_usable = False
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping) or set(attempt) != {
            "step_size",
            "mean_acceptance_probability",
            "finite",
            "engineering_health_failures",
            "usable",
            "seed",
        }:
            raise ValueError(f"{name} attempt field set is invalid")
        step = _finite_real(
            attempt.get("step_size"),
            name=f"{name} attempt step_size",
            positive=True,
        )
        finite = attempt.get("finite")
        usable = attempt.get("usable")
        if not isinstance(finite, bool) or not isinstance(usable, bool):
            raise ValueError(f"{name} attempt flags must be boolean")
        health = attempt.get("engineering_health_failures")
        if not isinstance(health, (tuple, list)):
            raise ValueError(f"{name} attempt health failures are invalid")
        health_codes = tuple(dict.fromkeys(str(item) for item in health))
        if any(not item for item in health_codes) or not set(health_codes).issubset(
            allowed_health
        ):
            raise ValueError(f"{name} attempt health failures are invalid")
        mean = attempt.get("mean_acceptance_probability")
        if finite:
            value = _finite_real(mean, name=f"{name} attempt acceptance")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} attempt acceptance is outside [0, 1]")
        elif mean is not None:
            raise ValueError(f"{name} nonfinite attempt carries acceptance")
        if usable is not (finite and not health_codes):
            raise ValueError(f"{name} attempt usable flag is inconsistent")
        _strict_seed(attempt.get("seed"), name=f"{name} attempt seed")
        if index == len(attempts) - 1:
            final_step = step
            final_usable = usable
    if not final_usable or not math.isclose(
        selected,
        float(final_step),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(f"{name} selection lacks final usable evidence")
    return selected


def _validate_operational_warmup_payload(warmup: Mapping[str, Any]) -> None:
    required_fields = {
        "schema",
        "status",
        "algorithm_id",
        "route_contract_version",
        "config",
        "initial_coordinate_signature",
        "final_coordinate_signature",
        "final_metric_signature",
        "final_epsilon",
        "trajectory_policy_signature",
        "reasonable_epsilon",
        "windows",
        "operational_metric_update_count",
        "every_update_used_by_later_transition",
        "private_start_bank",
        "seed_root",
        "target_scope",
        "target_status_trace_policy",
        "elapsed_s",
        "reports_posterior_convergence",
        "nonclaims",
    }
    optional_fields = {"metric_adaptation_status"}
    if (
        not required_fields <= set(warmup)
        or set(warmup) - required_fields - optional_fields
        or warmup.get("status") != "passed"
    ):
        raise ValueError("invalid operational warmup field set or status")
    if (
        warmup.get("algorithm_id") != OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
        or warmup.get("route_contract_version") != HMC_ROUTE_CONTRACT_VERSION
    ):
        raise ValueError("operational warmup algorithm route identity is invalid")
    for field in (
        "initial_coordinate_signature",
        "final_coordinate_signature",
        "final_metric_signature",
        "trajectory_policy_signature",
        "target_scope",
    ):
        if not str(warmup.get(field, "")):
            raise ValueError(f"operational warmup {field} is missing")
    _finite_real(warmup.get("final_epsilon"), name="warmup final_epsilon", positive=True)
    _finite_real(warmup.get("elapsed_s"), name="warmup elapsed_s")
    if float(warmup["elapsed_s"]) < 0.0:
        raise ValueError("warmup elapsed_s must be nonnegative")
    if warmup.get("every_update_used_by_later_transition") is not True:
        raise ValueError("warmup contains an unconsumed metric update")
    if warmup.get("reports_posterior_convergence") is not False:
        raise ValueError("warmup cannot report posterior convergence")
    if tuple(str(item) for item in warmup.get("nonclaims", ())) != (
        OPERATIONAL_WARMUP_NONCLAIMS
    ):
        raise ValueError("operational warmup nonclaims changed")
    _strict_seed(warmup.get("seed_root"), name="warmup seed_root")
    target_status_policy = str(warmup.get("target_status_trace_policy", ""))
    if target_status_policy not in {"none", "per_chain_step"}:
        raise ValueError("operational warmup target-status policy is invalid")
    initial_reasonable_step = _validate_reasonable_epsilon_payload(
        warmup.get("reasonable_epsilon"),
        name="warmup reasonable_epsilon",
    )

    config = warmup.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("operational warmup config is missing")
    required_config_fields = {
        "warmup_steps",
        "initial_buffer",
        "final_buffer",
        "first_window_size",
        "min_window_samples",
        "mass_shrinkage",
        "covariance_jitter",
        "eigenvalue_floor",
        "max_condition_number",
        "step_size_floor",
        "step_size_ceiling",
        "step_adaptation_rate",
    }
    optional_config_fields = {"mass_policy"}
    if not required_config_fields <= set(config) or (
        set(config) - required_config_fields - optional_config_fields
    ):
        raise ValueError("operational warmup config field set is invalid")
    integer_config = {
        name: _strict_scalar_integer(
            config.get(name),
            name=f"warmup config {name}",
        )
        for name in (
            "warmup_steps",
            "initial_buffer",
            "final_buffer",
            "first_window_size",
            "min_window_samples",
        )
    }
    optional_real_config = {}
    for name in ("eigenvalue_floor", "max_condition_number"):
        value = config.get(name)
        optional_real_config[name] = (
            None
            if value is None
            else _finite_real(value, name=f"warmup config {name}")
        )
    typed_config = WindowedMassAdaptationConfig(
        **integer_config,
        mass_shrinkage=_finite_real(
            config.get("mass_shrinkage"), name="warmup config mass_shrinkage"
        ),
        covariance_jitter=_finite_real(
            config.get("covariance_jitter"), name="warmup config covariance_jitter"
        ),
        step_size_floor=_finite_real(
            config.get("step_size_floor"), name="warmup config step_size_floor"
        ),
        step_size_ceiling=_finite_real(
            config.get("step_size_ceiling"), name="warmup config step_size_ceiling"
        ),
        step_adaptation_rate=_finite_real(
            config.get("step_adaptation_rate"),
            name="warmup config step_adaptation_rate",
        ),
        mass_policy=str(config.get("mass_policy", "windowed_adaptive")),
        **optional_real_config,
    )
    configured_steps = typed_config.warmup_steps
    windows = warmup.get("windows")
    if not isinstance(windows, (tuple, list)) or not windows:
        raise ValueError("operational warmup window ledger is missing")
    configured_windows = build_windowed_warmup_schedule(typed_config)
    if len(windows) != len(configured_windows):
        raise ValueError("operational warmup windows do not match configured schedule")

    window_fields = {
        "window",
        "transition_count_before_window",
        "transition_count_after_window",
        "coordinate_signature_used",
        "metric_signature_used",
        "epsilon_start",
        "epsilon_end",
        "mean_acceptance_probability",
        "binary_acceptance_rate",
        "native_divergence_status",
        "native_divergence_count",
        "target_status_trace_policy",
        "target_status_failure_count",
        "max_abs_log_accept_energy_proxy",
        "step_size_upper_bound",
        "maximum_bounded_next_step_size",
        "maximum_proposed_step_size",
        "maximum_consumed_step_size",
        "step_ceiling_hit_count",
        "metric_decision",
        "next_coordinate_signature",
        "next_metric_signature",
        "state_map_residual",
        "target_value_map_residual",
        "target_score_map_residual",
        "next_reasonable_epsilon",
        "dual_averaging_generation",
        "runner_generation",
        "runner_trace_count",
        "runtime_s",
        "raw_states_exposed",
    }
    nested_window_fields = {
        "index",
        "kind",
        "start",
        "end",
        "length",
        "update_mass",
    }
    expected_coordinate = str(warmup["initial_coordinate_signature"])
    expected_metric: str | None = None
    expected_transition = 0
    expected_epsilon: float | None = None
    applied_updates = 0
    for ordinal, raw_window in enumerate(windows):
        if not isinstance(raw_window, Mapping) or set(raw_window) != window_fields:
            raise ValueError("operational warmup window field set is invalid")
        window = raw_window.get("window")
        if not isinstance(window, Mapping) or set(window) != nested_window_fields:
            raise ValueError("operational warmup window identity is invalid")
        if dict(window) != configured_windows[ordinal].payload():
            raise ValueError("operational warmup windows do not match configured schedule")
        index = _strict_scalar_integer(window.get("index"), name="warmup window index")
        start = _strict_scalar_integer(window.get("start"), name="warmup window start")
        end = _strict_scalar_integer(window.get("end"), name="warmup window end")
        length = _strict_scalar_integer(window.get("length"), name="warmup window length")
        update_mass = window.get("update_mass")
        if not isinstance(update_mass, bool):
            raise ValueError("warmup window update_mass must be boolean")
        if (
            index != ordinal
            or start != expected_transition
            or end <= start
            or length != end - start
            or str(window.get("kind"))
            not in {"initial_fast", "slow", "final_fast"}
        ):
            raise ValueError("operational warmup window schedule is inconsistent")
        before = _strict_scalar_integer(
            raw_window.get("transition_count_before_window"),
            name="warmup transition_count_before_window",
        )
        after = _strict_scalar_integer(
            raw_window.get("transition_count_after_window"),
            name="warmup transition_count_after_window",
        )
        if before != start or after != end:
            raise ValueError("operational warmup transition counts are discontinuous")
        coordinate = str(raw_window.get("coordinate_signature_used", ""))
        metric = str(raw_window.get("metric_signature_used", ""))
        if coordinate != expected_coordinate or not metric:
            raise ValueError("operational warmup used a stale coordinate signature")
        if expected_metric is None:
            expected_metric = metric
        elif metric != expected_metric:
            raise ValueError("operational warmup used a stale metric signature")

        epsilon_start = _finite_real(
            raw_window.get("epsilon_start"),
            name="warmup epsilon_start",
            positive=True,
        )
        epsilon_end = _finite_real(
            raw_window.get("epsilon_end"),
            name="warmup epsilon_end",
            positive=True,
        )
        if expected_epsilon is not None and not math.isclose(
            epsilon_start,
            expected_epsilon,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise ValueError("operational warmup epsilon handoff is discontinuous")
        if ordinal == 0 and not math.isclose(
            epsilon_start,
            initial_reasonable_step,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("warmup initial epsilon does not match its bracket")
        for field in ("mean_acceptance_probability", "binary_acceptance_rate"):
            value = _finite_real(raw_window.get(field), name=f"warmup {field}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"warmup {field} must lie inside [0, 1]")
        proxy = _finite_real(
            raw_window.get("max_abs_log_accept_energy_proxy"),
            name="warmup log-accept proxy",
        )
        if proxy < 0.0:
            raise ValueError("warmup log-accept proxy must be nonnegative")
        step_upper_bound = _finite_real(
            raw_window.get("step_size_upper_bound"),
            name="warmup step_size_upper_bound",
            positive=True,
        )
        maximum_bounded = _finite_real(
            raw_window.get("maximum_bounded_next_step_size"),
            name="warmup maximum_bounded_next_step_size",
            positive=True,
        )
        maximum_proposed = _finite_real(
            raw_window.get("maximum_proposed_step_size"),
            name="warmup maximum_proposed_step_size",
            positive=True,
        )
        maximum_consumed = _finite_real(
            raw_window.get("maximum_consumed_step_size"),
            name="warmup maximum_consumed_step_size",
            positive=True,
        )
        if maximum_bounded > step_upper_bound * (1.0 + 1.0e-12):
            raise ValueError("warmup bounded step exceeds its ceiling")
        if maximum_consumed > step_upper_bound * (1.0 + 1.0e-12):
            raise ValueError("warmup consumed step exceeds its ceiling")
        ceiling_hits = _strict_scalar_integer(
            raw_window.get("step_ceiling_hit_count"),
            name="warmup step_ceiling_hit_count",
        )
        if ceiling_hits < 0:
            raise ValueError("warmup step ceiling hit count must be nonnegative")
        divergence_status = str(raw_window.get("native_divergence_status", ""))
        divergence_count = raw_window.get("native_divergence_count")
        if divergence_status == "available":
            normalized_divergence = _strict_scalar_integer(
                divergence_count,
                name="warmup native_divergence_count",
            )
            if normalized_divergence < 0:
                raise ValueError("warmup native divergence count must be nonnegative")
        elif divergence_status in {"not_exposed_by_kernel", "not_collected"}:
            if divergence_count is not None:
                raise ValueError("unavailable warmup divergence cannot carry a count")
        else:
            raise ValueError("warmup native divergence provenance is invalid")
        if str(raw_window.get("target_status_trace_policy", "")) != target_status_policy:
            raise ValueError("warmup window target-status policy is inconsistent")
        status_failure_count = raw_window.get("target_status_failure_count")
        if target_status_policy == "none":
            if status_failure_count is not None:
                raise ValueError("disabled warmup target status carries a count")
        else:
            status_count = _strict_scalar_integer(
                status_failure_count,
                name="warmup target_status_failure_count",
            )
            if status_count != 0:
                raise ValueError("passed warmup carries a target-status failure")
        for field in ("state_map_residual", "runtime_s"):
            value = _finite_real(raw_window.get(field), name=f"warmup {field}")
            if value < 0.0:
                raise ValueError(f"warmup {field} must be nonnegative")
        for field in ("target_value_map_residual", "target_score_map_residual"):
            value = raw_window.get(field)
            if value is not None:
                residual = _finite_real(value, name=f"warmup {field}")
                if residual < 0.0 or residual > 1.0e-10:
                    raise ValueError(f"warmup {field} violates the map invariant")
        generation = _strict_scalar_integer(
            raw_window.get("dual_averaging_generation"),
            name="warmup dual_averaging_generation",
        )
        runner_generation = _strict_scalar_integer(
            raw_window.get("runner_generation"),
            name="warmup runner_generation",
        )
        if generation != applied_updates or runner_generation != applied_updates:
            raise ValueError("warmup adaptation generation is inconsistent")
        trace_count = raw_window.get("runner_trace_count")
        if trace_count is not None and _strict_scalar_integer(
            trace_count,
            name="warmup runner_trace_count",
        ) <= 0:
            raise ValueError("warmup runner_trace_count must be positive")
        if raw_window.get("raw_states_exposed") is not False:
            raise ValueError("warmup window exposes raw states")

        decision = raw_window.get("metric_decision")
        next_coordinate = raw_window.get("next_coordinate_signature")
        next_metric = raw_window.get("next_metric_signature")
        next_reasonable = raw_window.get("next_reasonable_epsilon")
        update_applied = False
        if decision is not None:
            if not isinstance(decision, Mapping):
                raise ValueError("warmup metric decision is malformed")
            outcome = str(decision.get("outcome", ""))
            update_applied = decision.get("update_applied") is True
            expected_update = outcome in {"dense_update", "diagonal_fallback"}
            if update_applied is not expected_update:
                raise ValueError("warmup metric decision role is inconsistent")
            if outcome not in {
                "dense_update",
                "diagonal_fallback",
                "no_update_insufficient_metric_evidence",
                "candidate_metric_rejected",
            }:
                raise ValueError("warmup metric decision outcome is invalid")
        if update_applied:
            if not update_mass or ordinal + 1 >= len(windows):
                raise ValueError("warmup metric update lacks a later transition")
            if not str(next_coordinate or "") or not str(next_metric or ""):
                raise ValueError("warmup metric update lacks next signatures")
            if (
                not isinstance(next_reasonable, Mapping)
                or next_reasonable.get("status")
                not in {"passed", "externally_qualified"}
            ):
                raise ValueError("warmup metric update lacks epsilon rebracketing")
            next_step = _validate_reasonable_epsilon_payload(
                next_reasonable,
                name="warmup next reasonable epsilon",
            )
            expected_coordinate = str(next_coordinate)
            expected_metric = str(next_metric)
            epsilon_end = next_step
            applied_updates += 1
        elif any(item is not None for item in (next_coordinate, next_metric, next_reasonable)):
            raise ValueError("warmup no-update window carries a false handoff")
        expected_transition = end
        expected_epsilon = epsilon_end

    reported_updates = _strict_scalar_integer(
        warmup.get("operational_metric_update_count"),
        name="operational_metric_update_count",
    )
    if reported_updates != applied_updates:
        raise ValueError("operational warmup metric-update count is inconsistent")
    expected_adaptation_status = (
        "metric_updated" if applied_updates > 0 else "no_metric_update"
    )
    reported_adaptation_status = warmup.get("metric_adaptation_status")
    if (
        reported_adaptation_status is not None
        and reported_adaptation_status != expected_adaptation_status
    ):
        raise ValueError("operational warmup adaptation status is inconsistent")
    if expected_transition != configured_steps:
        raise ValueError("operational warmup did not execute its configured transitions")
    if (
        str(warmup["final_coordinate_signature"]) != expected_coordinate
        or str(warmup["final_metric_signature"]) != expected_metric
        or not math.isclose(
            float(warmup["final_epsilon"]),
            float(expected_epsilon),
            rel_tol=1.0e-12,
            abs_tol=0.0,
        )
    ):
        raise ValueError("operational warmup final kernel lineage is inconsistent")


def transition_ledger_payload(
    transitions: Sequence[HMCTuningTransition],
) -> Mapping[str, Any]:
    records = tuple(transitions)
    if not records:
        raise ValueError("transition ledger requires at least one transition")
    payload = {
        "schema": _TRANSITION_SCHEMA,
        "records": tuple(record.payload() for record in records),
        "record_count": len(records),
    }
    return {**payload, "ledger_sha256": canonical_sha256(payload)}


def build_hmc_tuning_engineering_artifact(
    *,
    evidence_purpose: str,
    configured_attempt_slots: int,
    warmup_payload: Mapping[str, Any],
    kernel_state_payload: Mapping[str, Any],
    start_bank_payload: Mapping[str, Any],
    trajectory_handoff: Mapping[str, Any],
    acceptance_evidence_payloads: Sequence[Mapping[str, Any]],
    transition_ledger: Mapping[str, Any],
    seed_domains: Mapping[str, Any],
    terminal_state: str,
    repair_loop_validated: bool,
    old_v1_compatibility: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    purpose = str(evidence_purpose)
    if purpose not in {"seam_execution_only", "repair_loop_validation"}:
        raise ValueError("unsupported evidence purpose")
    slots = _strict_scalar_integer(
        configured_attempt_slots,
        name="configured_attempt_slots",
    )
    if slots <= 0:
        raise ValueError("configured_attempt_slots must be positive")
    if not isinstance(repair_loop_validated, bool):
        raise ValueError("repair_loop_validated must be boolean")
    validated = repair_loop_validated
    if purpose == "seam_execution_only" and validated:
        raise ValueError("seam_execution_only cannot claim repair-loop validation")
    if purpose == "repair_loop_validation" and slots < 2:
        raise ValueError("repair-loop validation requires at least two attempt slots")
    core = {
        "schema": _ARTIFACT_SCHEMA,
        "algorithm_id": warmup_payload.get("algorithm_id"),
        "route_contract_version": warmup_payload.get("route_contract_version"),
        "evidence_purpose": purpose,
        "configured_attempt_slots": slots,
        "warmup": dict(warmup_payload),
        "kernel_state": dict(kernel_state_payload),
        "private_start_bank": dict(start_bank_payload),
        "trajectory_handoff": dict(trajectory_handoff),
        "acceptance_evidence": tuple(
            dict(item) for item in acceptance_evidence_payloads
        ),
        "transition_ledger": dict(transition_ledger),
        "seed_domains": dict(seed_domains),
        "terminal_state": str(terminal_state),
        "repair_loop_validated": validated,
        "old_v1_compatibility": (
            None if old_v1_compatibility is None else dict(old_v1_compatibility)
        ),
        "raw_start_bank_exposed": False,
        "raw_states_exposed": False,
        "raw_samples_exposed": False,
        "reports_posterior_convergence": False,
        "reports_sampler_superiority": False,
        "reports_default_readiness": False,
        "reports_gpu_or_xla_readiness": False,
        "nonclaims": TUNING_ARTIFACT_NONCLAIMS,
    }
    validate_hmc_tuning_engineering_artifact(core, require_hash=False)
    return {**core, "artifact_sha256": canonical_sha256(core)}


def validate_hmc_tuning_engineering_artifact(
    payload: Mapping[str, Any],
    *,
    require_hash: bool = True,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != _ARTIFACT_SCHEMA:
        raise ValueError("invalid HMC tuning artifact schema")
    expected_fields = {
        "schema",
        "algorithm_id",
        "route_contract_version",
        "evidence_purpose",
        "configured_attempt_slots",
        "warmup",
        "kernel_state",
        "private_start_bank",
        "trajectory_handoff",
        "acceptance_evidence",
        "transition_ledger",
        "seed_domains",
        "terminal_state",
        "repair_loop_validated",
        "old_v1_compatibility",
        "raw_start_bank_exposed",
        "raw_states_exposed",
        "raw_samples_exposed",
        "reports_posterior_convergence",
        "reports_sampler_superiority",
        "reports_default_readiness",
        "reports_gpu_or_xla_readiness",
        "nonclaims",
    }
    if require_hash:
        expected_fields.add("artifact_sha256")
    if set(payload) != expected_fields:
        raise ValueError("HMC tuning artifact field set is inconsistent")
    purpose = str(payload.get("evidence_purpose", ""))
    slots = _strict_scalar_integer(
        payload.get("configured_attempt_slots"),
        name="configured_attempt_slots",
    )
    raw_validated = payload.get("repair_loop_validated")
    if not isinstance(raw_validated, bool):
        raise ValueError("repair_loop_validated must be boolean")
    validated = raw_validated
    if purpose not in {"seam_execution_only", "repair_loop_validation"}:
        raise ValueError("invalid evidence purpose")
    if slots <= 0:
        raise ValueError("invalid attempt-slot count")
    if purpose == "seam_execution_only" and validated:
        raise ValueError("one-purpose seam artifact cannot validate the repair loop")
    if purpose == "repair_loop_validation" and slots < 2:
        raise ValueError("repair-loop artifact lacks a reserved verification slot")

    warmup = payload.get("warmup")
    if not isinstance(warmup, Mapping) or warmup.get("schema") != (
        "bayesfilter.hmc_operational_windowed_warmup.v2"
    ):
        raise ValueError("invalid operational warmup payload")
    _validate_operational_warmup_payload(warmup)
    if (
        payload.get("algorithm_id") != warmup.get("algorithm_id")
        or payload.get("route_contract_version")
        != warmup.get("route_contract_version")
    ):
        raise ValueError("artifact/warmup algorithm route lineage mismatch")

    kernel = payload.get("kernel_state")
    if not isinstance(kernel, Mapping) or kernel.get("schema") != _KERNEL_SCHEMA:
        raise ValueError("invalid kernel-state payload")
    for field in (
        "coordinate_signature",
        "metric_signature",
        "trajectory_signature",
        "epsilon_context_signature",
        "canonical_theta_signature",
    ):
        if not str(kernel.get(field, "")):
            raise ValueError(f"kernel-state {field} is missing")
    if kernel.get("raw_state_values_exposed") is not False:
        raise ValueError("kernel-state privacy contract failed")
    allowed_kernel_fields = {
        "schema",
        "coordinate_signature",
        "metric_signature",
        "trajectory_signature",
        "epsilon_context_signature",
        "epsilon",
        "adaptation_generation",
        "seed_lineage",
        "evidence_status",
        "canonical_theta_signature",
        "raw_state_values_exposed",
    }
    if set(kernel) != allowed_kernel_fields:
        raise ValueError("kernel-state summary contains unexpected fields")
    epsilon = kernel.get("epsilon")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, Real)
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0.0
    ):
        raise ValueError("kernel-state epsilon must be positive and finite")
    adaptation_generation = _strict_scalar_integer(
        kernel.get("adaptation_generation"),
        name="kernel-state adaptation_generation",
    )
    if adaptation_generation < 0:
        raise ValueError("kernel-state adaptation_generation must be nonnegative")
    _strict_seed(kernel.get("seed_lineage"), name="kernel-state seed_lineage")
    if not str(kernel.get("evidence_status", "")):
        raise ValueError("kernel-state evidence_status is missing")
    if kernel["coordinate_signature"] != warmup.get("final_coordinate_signature"):
        raise ValueError("kernel/warmup coordinate signature mismatch")
    if kernel["metric_signature"] != warmup.get("final_metric_signature"):
        raise ValueError("kernel/warmup metric signature mismatch")
    bank = payload.get("private_start_bank")
    warmup_bank = warmup.get("private_start_bank")
    if not isinstance(bank, Mapping) or bank.get("schema") != _START_BANK_SCHEMA:
        raise ValueError("private start-bank lineage or privacy mismatch")
    bank_count = _strict_scalar_integer(
        bank.get("count"),
        name="private start-bank count",
    )
    _strict_seed(bank.get("seed_root"), name="private start-bank seed_root")
    if (
        not isinstance(warmup_bank, Mapping)
        or bank.get("source_signature") != warmup_bank.get("signature")
        or not str(bank.get("signature", ""))
        or bank_count != 4
        or bank.get("coordinate_signature") != kernel["coordinate_signature"]
        or bank.get("raw_values_exposed") is not False
        or bank.get("paths_exposed") is not False
    ):
        raise ValueError("private start-bank lineage or privacy mismatch")
    allowed_bank_fields = {
        "schema",
        "signature",
        "source_signature",
        "count",
        "coordinate_signature",
        "seed_root",
        "raw_values_exposed",
        "paths_exposed",
    }
    if set(bank) != allowed_bank_fields:
        raise ValueError("private start-bank summary contains unexpected fields")

    handoff = payload.get("trajectory_handoff")
    allowed_handoff_fields = {
        "schema",
        "warmup_trajectory_signature",
        "final_trajectory_signature",
        "warmup_num_leapfrog_steps",
        "selected_num_leapfrog_steps",
        "max_leapfrog_steps",
        "selection_signature",
        "candidate_signature",
        "exact_l_retune_signature",
        "exact_l_retune_seed",
        "coordinate_signature",
        "metric_signature",
        "start_bank_signature",
        "exact_l_retuned",
    }
    if (
        not isinstance(handoff, Mapping)
        or handoff.get("schema") != _TRAJECTORY_HANDOFF_SCHEMA
        or set(handoff) != allowed_handoff_fields
    ):
        raise ValueError("invalid trajectory handoff payload")
    warmup_l = _strict_scalar_integer(
        handoff.get("warmup_num_leapfrog_steps"),
        name="warmup_num_leapfrog_steps",
    )
    selected_l = _strict_scalar_integer(
        handoff.get("selected_num_leapfrog_steps"),
        name="selected_num_leapfrog_steps",
    )
    maximum_l = _strict_scalar_integer(
        handoff.get("max_leapfrog_steps"),
        name="max_leapfrog_steps",
    )
    retune_seed = _strict_seed(
        handoff.get("exact_l_retune_seed"),
        name="exact_l_retune_seed",
    )
    expected_warmup_trajectory = WarmupTrajectoryPolicy(warmup_l, maximum_l).signature
    expected_final_trajectory = WarmupTrajectoryPolicy(selected_l, maximum_l).signature
    if (
        warmup_l <= 0
        or selected_l <= 0
        or selected_l > maximum_l
        or handoff.get("warmup_trajectory_signature") != expected_warmup_trajectory
        or handoff.get("warmup_trajectory_signature")
        != warmup.get("trajectory_policy_signature")
        or handoff.get("final_trajectory_signature") != expected_final_trajectory
        or handoff.get("final_trajectory_signature") != kernel["trajectory_signature"]
        or handoff.get("coordinate_signature") != kernel["coordinate_signature"]
        or handoff.get("metric_signature") != kernel["metric_signature"]
        or handoff.get("start_bank_signature") != bank["signature"]
        or handoff.get("exact_l_retuned") is not True
        or len(retune_seed) != 2
        or not str(handoff.get("selection_signature", ""))
        or not str(handoff.get("candidate_signature", ""))
        or not str(handoff.get("exact_l_retune_signature", ""))
    ):
        raise ValueError("trajectory handoff lineage or exact-L retune is invalid")

    evidence_payloads = payload.get("acceptance_evidence")
    if not isinstance(evidence_payloads, (tuple, list)) or not evidence_payloads:
        raise ValueError("artifact requires acceptance evidence")
    evidence = tuple(
        hmc_acceptance_evidence_from_payload(item) for item in evidence_payloads
    )

    ledger = payload.get("transition_ledger")
    if not isinstance(ledger, Mapping) or ledger.get("schema") != _TRANSITION_SCHEMA:
        raise ValueError("invalid transition ledger")
    records = ledger.get("records")
    if not isinstance(records, (tuple, list)):
        raise ValueError("transition ledger count mismatch")
    record_count = _strict_scalar_integer(
        ledger.get("record_count"),
        name="transition ledger record_count",
    )
    if record_count != len(records):
        raise ValueError("transition ledger count mismatch")
    ledger_core = {
        "schema": ledger["schema"],
        "records": tuple(records),
        "record_count": len(records),
    }
    if ledger.get("ledger_sha256") != canonical_sha256(ledger_core):
        raise ValueError("transition ledger hash mismatch")
    transitions = tuple(HMCTuningTransition(**dict(item)) for item in records)
    if transitions[0].source != "initialized":
        raise ValueError("transition ledger must begin at initialized")
    if any(
        left.target != right.source
        for left, right in zip(transitions, transitions[1:])
    ):
        raise ValueError("transition ledger is not contiguous")
    if transitions[-1].target != str(payload.get("terminal_state")):
        raise ValueError("transition ledger terminal-state mismatch")
    if validated:
        edges = tuple((item.source, item.target) for item in transitions)
        required_edges = (
            ("verifying", "repair_required"),
            ("repair_required", "step_repaired"),
            ("step_repaired", "verifying"),
        )
        edge_indices = []
        search_start = 0
        for edge in required_edges:
            try:
                index = edges.index(edge, search_start)
            except ValueError as exc:
                raise ValueError(
                    "repair-loop validation lacks verify-repair-reverify transitions"
                ) from exc
            edge_indices.append(index)
            search_start = index + 1
        repair_edges = tuple(transitions[index] for index in edge_indices)
        signature_fields = (
            "coordinate_signature",
            "metric_signature",
            "trajectory_signature",
        )
        for field in signature_fields:
            values = tuple(getattr(item, field) for item in repair_edges)
            if any(value is None for value in values) or len(set(values)) != 1:
                raise ValueError("step repair changed or omitted a frozen signature")
        if len(evidence) < 2:
            raise ValueError("repair-loop validation requires pre/post repair evidence")
        directional_indices = tuple(
            index
            for index, item in enumerate(evidence)
            if item.evidence_validity == "valid"
            and item.acceptance_decision
            in {"repair_step_lower", "repair_step_higher"}
            and item.repair_direction is not None
        )
        if not directional_indices or directional_indices[0] >= len(evidence) - 1:
            raise ValueError(
                "repair-loop validation requires directional evidence before re-verification"
            )

    seed_domains = payload.get("seed_domains")
    if not isinstance(seed_domains, Mapping):
        raise ValueError("seed domains are missing")
    required_domains = {
        "warmup",
        "candidate_selection",
        "exact_final_l_epsilon_tune",
        "independent_final_verification",
        "repair_verification",
        "evidence_extension",
    }
    if set(seed_domains) != required_domains:
        raise ValueError("seed-domain ledger is incomplete")
    normalized_seeds = tuple(
        _strict_seed(seed_domains[key], name=f"seed domain {key}")
        for key in sorted(seed_domains)
    )
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seed domains must contain distinct two-integer seeds")
    if (
        _strict_seed(
            seed_domains["exact_final_l_epsilon_tune"],
            name="seed domain exact_final_l_epsilon_tune",
        )
        != retune_seed
    ):
        raise ValueError("trajectory handoff exact-L seed-domain mismatch")

    if any(payload.get(field) is not False for field in (
        "raw_start_bank_exposed",
        "raw_states_exposed",
        "raw_samples_exposed",
        "reports_posterior_convergence",
        "reports_sampler_superiority",
        "reports_default_readiness",
        "reports_gpu_or_xla_readiness",
    )):
        raise ValueError("artifact privacy or nonclaim contract failed")
    nonclaims = tuple(str(item) for item in payload.get("nonclaims", ()))
    if nonclaims != TUNING_ARTIFACT_NONCLAIMS:
        raise ValueError("artifact nonclaims changed")

    if require_hash:
        expected_hash = str(payload.get("artifact_sha256", ""))
        core = {key: value for key, value in payload.items() if key != "artifact_sha256"}
        if expected_hash != canonical_sha256(core):
            raise ValueError("HMC tuning artifact hash mismatch")
    return {
        "artifact_sha256": payload.get("artifact_sha256"),
        "evidence_validities": tuple(
            item.evidence_validity for item in evidence
        ),
        "evidence_decisions": tuple(
            item.acceptance_decision for item in evidence
        ),
        "promotion_eligibility": tuple(
            item.promotion_eligible for item in evidence
        ),
        "transition_count": len(transitions),
        "terminal_state": transitions[-1].target,
        "repair_loop_validated": validated,
        "old_v1_compatibility_present": payload.get("old_v1_compatibility") is not None,
    }


def load_and_replay_hmc_tuning_artifact(
    path: str | os.PathLike[str],
) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="ascii"))
    if payload.get("schema") == _LEGACY_ARTIFACT_SCHEMA:
        return _legacy_hmc_tuning_artifact_migration_view(payload)
    return validate_hmc_tuning_engineering_artifact(payload)


def _legacy_hmc_tuning_artifact_migration_view(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Read a hash-valid v2 envelope without granting v3 repair authority."""

    if not isinstance(payload, Mapping) or payload.get("schema") != (
        _LEGACY_ARTIFACT_SCHEMA
    ):
        raise ValueError("invalid legacy HMC tuning artifact schema")
    expected_hash = str(payload.get("artifact_sha256", ""))
    core = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    if expected_hash != canonical_sha256(core):
        raise ValueError("legacy HMC tuning artifact hash mismatch")
    evidence_payloads = payload.get("acceptance_evidence")
    if not isinstance(evidence_payloads, (tuple, list)) or not evidence_payloads:
        raise ValueError("legacy artifact requires acceptance evidence")
    migration_views = tuple(
        hmc_acceptance_evidence_v2_migration_view(item)
        for item in evidence_payloads
    )
    for field in (
        "raw_start_bank_exposed",
        "raw_states_exposed",
        "raw_samples_exposed",
        "reports_posterior_convergence",
        "reports_sampler_superiority",
        "reports_default_readiness",
        "reports_gpu_or_xla_readiness",
    ):
        if payload.get(field) is not False:
            raise ValueError("legacy artifact privacy or nonclaim contract failed")
    return {
        "schema": "bayesfilter.hmc_tuning_engineering_artifact_v2_migration_view.v1",
        "source_schema": _LEGACY_ARTIFACT_SCHEMA,
        "source_artifact_sha256": expected_hash,
        "historical_envelope_integrity": "hash_and_evidence_contract_valid",
        "evidence_migration_views": migration_views,
        "operational_authority": False,
        "repair_loop_validated_under_v4": False,
        "source_payload_mutated": False,
    }


@dataclass(frozen=True)
class KillableChildSpec:
    command: tuple[str, ...]
    timeout_s: float
    closeout_path: Path
    child_artifact_path: Path | None = None
    cwd: Path | None = None
    environment: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        command = tuple(str(item) for item in self.command)
        timeout = float(self.timeout_s)
        if not command or timeout <= 0.0:
            raise ValueError("killable child requires a command and positive timeout")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "timeout_s", timeout)
        object.__setattr__(self, "closeout_path", Path(self.closeout_path))
        object.__setattr__(
            self,
            "child_artifact_path",
            None if self.child_artifact_path is None else Path(self.child_artifact_path),
        )
        object.__setattr__(self, "cwd", None if self.cwd is None else Path(self.cwd))


def run_killable_child(spec: KillableChildSpec) -> Mapping[str, Any]:
    if not isinstance(spec, KillableChildSpec):
        raise TypeError("spec must be KillableChildSpec")
    environment = os.environ.copy()
    if spec.environment is not None:
        environment.update({str(key): str(value) for key, value in spec.environment.items()})
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    started_wall = time.time()
    started_monotonic = time.monotonic()
    process = subprocess.Popen(
        spec.command,
        cwd=None if spec.cwd is None else str(spec.cwd),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=spec.timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    elapsed = max(0.0, time.monotonic() - started_monotonic)
    child_artifact_exists = bool(
        spec.child_artifact_path is not None and spec.child_artifact_path.is_file()
    )
    child_artifact_hash = (
        file_sha256(spec.child_artifact_path) if child_artifact_exists else None
    )
    classification = (
        "hard_timeout"
        if timed_out
        else "completed"
        if process.returncode == 0
        else "child_failed"
    )
    core = {
        "schema": _TIMEOUT_SCHEMA,
        "classification": classification,
        "command": spec.command,
        "cwd": None if spec.cwd is None else str(spec.cwd),
        "timeout_s": spec.timeout_s,
        "started_unix_s": started_wall,
        "elapsed_s": elapsed,
        "returncode": int(process.returncode),
        "parent_finalized": True,
        "gpu_intentionally_hidden": True,
        "child_artifact_path": (
            None if spec.child_artifact_path is None else str(spec.child_artifact_path)
        ),
        "child_artifact_exists": child_artifact_exists,
        "child_artifact_sha256": child_artifact_hash,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "stdout_retained": False,
        "stderr_retained": False,
        "reports_posterior_convergence": False,
        "reports_gpu_or_xla_readiness": False,
        "nonclaims": TUNING_ARTIFACT_NONCLAIMS,
    }
    closeout = {**core, "closeout_sha256": canonical_sha256(core)}
    atomic_write_json(spec.closeout_path, closeout)
    return closeout


def validate_killable_child_closeout(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != _TIMEOUT_SCHEMA:
        raise ValueError("invalid killable-child closeout schema")
    if payload.get("classification") not in {"completed", "child_failed", "hard_timeout"}:
        raise ValueError("invalid killable-child classification")
    if payload.get("parent_finalized") is not True or payload.get("gpu_intentionally_hidden") is not True:
        raise ValueError("killable-child closeout was not parent-finalized CPU-only evidence")
    core = {key: value for key, value in payload.items() if key != "closeout_sha256"}
    if payload.get("closeout_sha256") != canonical_sha256(core):
        raise ValueError("killable-child closeout hash mismatch")
