"""Testing diagnostics for Fixed-SGQF results.

These helpers are testing-only. They normalize the Fixed-SGQF value and score
result containers into a small comparable vocabulary used by audit and
integration tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from bayesfilter.nonlinear.fixed_sgqf_derivatives_tf import TFFixedSGQFScoreResult
from bayesfilter.nonlinear.fixed_sgqf_tf import TFFixedSGQFStepFailure, TFFixedSGQFValueResult


@dataclass(frozen=True)
class FixedSGQFDiagnosticSnapshot:
    """Comparable scalar diagnostics for Fixed-SGQF value and score results."""

    branch_hash: str
    runtime_mode: str
    rule_family: str
    cloud_point_count: int
    weight_total: float
    negative_weight_count: int
    accepted_steps: int
    observation_preprocessing: str
    initial_condition_policy: str
    failure_record_policy: str
    factor_branch: str
    additive_noise_policy: str
    veto_policy: str
    predictive_epsilon: float
    innovation_epsilon: float
    failure_stage: str | None
    failure_reason: str | None
    failure_time_index: int | None
    derivative_method: str | None
    same_branch_signature: tuple[str, str, int] | None


@dataclass(frozen=True)
class FixedSGQFBranchSummary:
    """Small eager-mode summary of Fixed-SGQF branch outcomes."""

    total_count: int
    ok_count: int
    weight_sum_failure_count: int
    previous_covariance_failure_count: int
    predictive_covariance_failure_count: int
    innovation_covariance_failure_count: int
    carried_covariance_failure_count: int
    same_scalar_branch_mismatch_count: int
    failure_labels: tuple[str, ...]


def fixed_sgqf_diagnostic_snapshot(
    result: TFFixedSGQFValueResult | TFFixedSGQFScoreResult,
) -> FixedSGQFDiagnosticSnapshot:
    diagnostics = result.diagnostics
    failure_stage = diagnostics.get("failure_stage")
    failure_reason = diagnostics.get("failure_reason")
    failure_time_index = diagnostics.get("failure_time_index")
    same_branch_signature = diagnostics.get("same_branch_signature")
    return FixedSGQFDiagnosticSnapshot(
        branch_hash=str(diagnostics["branch_hash"]),
        runtime_mode=str(diagnostics["runtime_mode"]),
        rule_family=str(diagnostics["rule_family"]),
        cloud_point_count=_to_int(diagnostics["cloud_point_count"]),
        weight_total=_to_float(diagnostics["weight_total"]),
        negative_weight_count=_to_int(diagnostics["negative_weight_count"]),
        accepted_steps=_to_int(diagnostics["accepted_steps"]),
        observation_preprocessing=str(diagnostics["observation_preprocessing"]),
        initial_condition_policy=str(diagnostics["initial_condition_policy"]),
        failure_record_policy=str(diagnostics["failure_record_policy"]),
        factor_branch=str(diagnostics["factor_branch"]),
        additive_noise_policy=str(diagnostics["additive_noise_policy"]),
        veto_policy=str(diagnostics["veto_policy"]),
        predictive_epsilon=_to_float(diagnostics["predictive_epsilon"]),
        innovation_epsilon=_to_float(diagnostics["innovation_epsilon"]),
        failure_stage=None if failure_stage is None else str(failure_stage),
        failure_reason=None if failure_reason is None else str(failure_reason),
        failure_time_index=None if failure_time_index is None else _to_int(failure_time_index),
        derivative_method=None if "derivative_method" not in diagnostics else str(diagnostics["derivative_method"]),
        same_branch_signature=None if same_branch_signature is None else tuple(same_branch_signature),
    )


def fixed_sgqf_failure_label(failure: TFFixedSGQFStepFailure | None) -> str:
    if failure is None:
        return "accepted"
    return f"{failure.stage}:{failure.reason}"


def fixed_sgqf_branch_summary(results: list[TFFixedSGQFValueResult | TFFixedSGQFScoreResult]) -> FixedSGQFBranchSummary:
    labels: list[str] = []
    ok_count = 0
    counters = {
        "cloud:weight_sum_failure": 0,
        "previous_covariance:positive_definiteness_veto": 0,
        "previous_covariance:cholesky_failure": 0,
        "predictive_covariance:positive_definiteness_veto": 0,
        "predictive_covariance:cholesky_failure": 0,
        "innovation_covariance:positive_definiteness_veto": 0,
        "innovation_covariance:cholesky_failure": 0,
        "carried_covariance:positive_definiteness_veto": 0,
        "carried_covariance:cholesky_failure": 0,
        "branch_signature:same_scalar_branch_mismatch": 0,
    }
    for result in results:
        label = fixed_sgqf_failure_label(result.failure)
        if label == "accepted":
            ok_count += 1
        else:
            counters[label] = counters.get(label, 0) + 1
            if label not in labels and len(labels) < 5:
                labels.append(label)
    return FixedSGQFBranchSummary(
        total_count=len(results),
        ok_count=ok_count,
        weight_sum_failure_count=counters.get("cloud:weight_sum_failure", 0),
        previous_covariance_failure_count=(
            counters.get("previous_covariance:positive_definiteness_veto", 0)
            + counters.get("previous_covariance:cholesky_failure", 0)
        ),
        predictive_covariance_failure_count=(
            counters.get("predictive_covariance:positive_definiteness_veto", 0)
            + counters.get("predictive_covariance:cholesky_failure", 0)
        ),
        innovation_covariance_failure_count=(
            counters.get("innovation_covariance:positive_definiteness_veto", 0)
            + counters.get("innovation_covariance:cholesky_failure", 0)
        ),
        carried_covariance_failure_count=(
            counters.get("carried_covariance:positive_definiteness_veto", 0)
            + counters.get("carried_covariance:cholesky_failure", 0)
        ),
        same_scalar_branch_mismatch_count=counters.get("branch_signature:same_scalar_branch_mismatch", 0),
        failure_labels=tuple(labels),
    )


def _to_float(value: object) -> float:
    return float(tf.convert_to_tensor(value, dtype=tf.float64).numpy())


def _to_int(value: object) -> int:
    return int(tf.convert_to_tensor(value).numpy())
