"""Pure work-budget contracts for operational HMC kernel tuning.

This module is deliberately independent of TensorFlow, TFP, and NumPy so a
launcher can validate the maximum statistical work before initializing an
accelerator.  Counts describe batched HMC transition steps; chain-expanded
work is reported separately.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


OPERATIONAL_HMC_BUDGET_POLICY_ID = (
    "bayesfilter.hmc_operational_statistical_work.v1"
)
OPERATIONAL_HMC_BUDGET_NONCLAIMS = (
    "statistical-work allocation and accounting only",
    "compatibility counts are not calibrated optima",
    "no posterior convergence claim",
    "no sampler superiority claim",
    "no default-readiness claim",
    "no GPU or XLA readiness claim",
)
_PUBLIC_MANIFEST_SCHEMA = "bayesfilter.hmc_operational_work_manifest.v2"
_PRIVATE_MANIFEST_SCHEMA = "bayesfilter.hmc_operational_private_work_manifest.v1"
_RECONCILIATION_SCHEMA = "bayesfilter.hmc_operational_work_reconciliation.v1"
_ACCOUNTING_SCOPE_ID = (
    "operational_metric_adaptation_through_fresh_verification.v1"
)
_FORBIDDEN_PUBLIC_KEYS = {
    "candidate_identity",
    "candidate_payload",
    "candidate_values",
    "epsilon",
    "mass",
    "mass_matrix",
    "num_leapfrog_steps",
    "private_path",
    "sample",
    "samples",
    "seed",
    "state",
    "step_size",
}
_FORBIDDEN_PUBLIC_KEY_COMPONENTS = {
    "covariance",
    "epsilon",
    "factor",
    "mass",
    "position",
    "sample",
    "samples",
    "seed",
    "state",
}


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _strict_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _strict_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _assert_public_payload_safe(value: Any, *, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.lower()
            components = set(normalized.split("_"))
            if (
                normalized in _FORBIDDEN_PUBLIC_KEYS
                or components.intersection(_FORBIDDEN_PUBLIC_KEY_COMPONENTS)
            ):
                raise ValueError(f"public budget manifest exposes private field {path}.{key}")
            _assert_public_payload_safe(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_public_payload_safe(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class HMCOperationalStatisticalWorkPolicy:
    """Independent operational budgets; no field is derived from warmup size."""

    initial_candidate_results: int = 64
    candidate_burnin_steps: int = 16
    evidence_extension_checkpoints: tuple[int, ...] = ()
    exact_l_tune_adaptation_steps: int = 64
    fresh_verification_results: int = 64
    fresh_verification_burnin_steps: int = 16
    candidate_count_upper_bound: int = 3
    replications_per_candidate: int = 3
    exact_l_tune_result_steps: int = 4
    fresh_verification_starts_per_outer_attempt: int = 2
    chain_count: int = 4
    policy_id: str = OPERATIONAL_HMC_BUDGET_POLICY_ID

    def __post_init__(self) -> None:
        for name in (
            "initial_candidate_results",
            "candidate_burnin_steps",
            "exact_l_tune_adaptation_steps",
            "fresh_verification_results",
            "fresh_verification_burnin_steps",
            "candidate_count_upper_bound",
            "replications_per_candidate",
            "exact_l_tune_result_steps",
            "fresh_verification_starts_per_outer_attempt",
            "chain_count",
        ):
            object.__setattr__(
                self,
                name,
                _strict_positive_int(getattr(self, name), name=name),
            )
        checkpoints = tuple(
            _strict_positive_int(item, name="evidence extension checkpoint")
            for item in self.evidence_extension_checkpoints
        )
        if any(item <= self.initial_candidate_results for item in checkpoints):
            raise ValueError(
                "evidence extension checkpoints must exceed initial candidate results"
            )
        if any(left >= right for left, right in zip(checkpoints, checkpoints[1:])):
            raise ValueError("evidence extension checkpoints must strictly increase")
        if len(checkpoints) > 2:
            raise ValueError("at most two evidence extension checkpoints are supported")
        object.__setattr__(self, "evidence_extension_checkpoints", checkpoints)
        policy_id = str(self.policy_id)
        if not policy_id:
            raise ValueError("policy_id must be non-empty")
        object.__setattr__(self, "policy_id", policy_id)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_operational_statistical_work_policy.v1",
            "policy_id": self.policy_id,
            "initial_candidate_results": self.initial_candidate_results,
            "candidate_burnin_steps": self.candidate_burnin_steps,
            "evidence_extension_checkpoints": self.evidence_extension_checkpoints,
            "exact_l_tune_adaptation_steps": self.exact_l_tune_adaptation_steps,
            "fresh_verification_results": self.fresh_verification_results,
            "fresh_verification_burnin_steps": (
                self.fresh_verification_burnin_steps
            ),
            "candidate_count_upper_bound": self.candidate_count_upper_bound,
            "replications_per_candidate": self.replications_per_candidate,
            "exact_l_tune_result_steps": self.exact_l_tune_result_steps,
            "fresh_verification_starts_per_outer_attempt": (
                self.fresh_verification_starts_per_outer_attempt
            ),
            "chain_count": self.chain_count,
            "extensions_enabled": bool(self.evidence_extension_checkpoints),
            "roles": {
                "candidate_evidence": "kernel_handoff_screen",
                "exact_l_tune": "post_nomination_epsilon_adaptation",
                "fresh_verification": "promotion_or_repair_or_hard_veto",
            },
            "nonclaims": OPERATIONAL_HMC_BUDGET_NONCLAIMS,
        }

    @property
    def policy_hash(self) -> str:
        return _canonical_hash(self.payload())


def serious_metric_adaptation_schedule(
    *,
    target_dimension: int,
    outer_attempt_count: int,
    dimension_factor: int = 20,
    min_initial_budget: int = 1000,
    max_initial_budget: int = 5000,
    max_tune_budget: int = 10000,
) -> tuple[int, ...]:
    """Return the dimension-only serious warmup schedule used pre-runtime."""

    dimension = _strict_positive_int(target_dimension, name="target_dimension")
    attempts = _strict_positive_int(outer_attempt_count, name="outer_attempt_count")
    factor = _strict_positive_int(dimension_factor, name="dimension_factor")
    lower = _strict_positive_int(min_initial_budget, name="min_initial_budget")
    initial_cap = _strict_positive_int(max_initial_budget, name="max_initial_budget")
    tune_cap = _strict_positive_int(max_tune_budget, name="max_tune_budget")
    if lower > initial_cap or initial_cap > tune_cap:
        raise ValueError("serious metric budget caps are inconsistent")
    budget0 = min(initial_cap, max(lower, factor * dimension))
    return tuple(min(tune_cap, budget0 * (2**index)) for index in range(attempts))


def build_public_hmc_work_manifest(
    *,
    target_dimension: int,
    metric_adaptation_steps: Sequence[int],
    selection_attempts_per_outer_attempt: Sequence[int],
    max_leapfrog_steps: int,
    policy: HMCOperationalStatisticalWorkPolicy | None = None,
    algorithm_id: str = "operational_paired_fixed_trajectory_selection_v3",
    run_class: str = "serious",
) -> Mapping[str, Any]:
    """Build the conservative public upper bound before runtime initialization."""

    active = HMCOperationalStatisticalWorkPolicy() if policy is None else policy
    if not isinstance(active, HMCOperationalStatisticalWorkPolicy):
        raise TypeError("policy must be HMCOperationalStatisticalWorkPolicy")
    dimension = _strict_positive_int(target_dimension, name="target_dimension")
    leapfrog_cap = _strict_positive_int(max_leapfrog_steps, name="max_leapfrog_steps")
    metric = tuple(
        _strict_positive_int(item, name="metric adaptation step")
        for item in metric_adaptation_steps
    )
    selection_attempts = tuple(
        _strict_positive_int(item, name="selection attempts")
        for item in selection_attempts_per_outer_attempt
    )
    if not metric or len(metric) != len(selection_attempts):
        raise ValueError(
            "metric and selection-attempt schedules must have equal non-zero length"
        )
    if any(item > 5 for item in selection_attempts):
        raise ValueError("selection attempts per outer attempt are capped at five")
    algorithm = str(algorithm_id)
    classification = str(run_class)
    if not algorithm or not classification:
        raise ValueError("algorithm_id and run_class must be non-empty")

    selection_attempt_total = sum(selection_attempts)
    candidate_replication_slots = (
        active.candidate_count_upper_bound * active.replications_per_candidate
    )
    initial_candidate_transitions = (
        selection_attempt_total
        * candidate_replication_slots
        * (active.initial_candidate_results + active.candidate_burnin_steps)
    )
    extension_candidate_transitions = (
        selection_attempt_total
        * candidate_replication_slots
        * sum(
            checkpoint + active.candidate_burnin_steps
            for checkpoint in active.evidence_extension_checkpoints
        )
    )
    retune_starts_per_selection_attempt = (
        active.candidate_count_upper_bound
        * (1 + len(active.evidence_extension_checkpoints))
    )
    exact_l_tune_transitions = (
        selection_attempt_total
        * retune_starts_per_selection_attempt
        * (
            active.exact_l_tune_adaptation_steps
            + active.exact_l_tune_result_steps
        )
    )
    verification_start_count = (
        len(metric) * active.fresh_verification_starts_per_outer_attempt
    )
    fresh_verification_transitions = verification_start_count * (
        active.fresh_verification_results
        + active.fresh_verification_burnin_steps
    )
    metric_transitions = sum(metric)
    total_batched_transitions = sum(
        (
            metric_transitions,
            initial_candidate_transitions,
            extension_candidate_transitions,
            exact_l_tune_transitions,
            fresh_verification_transitions,
        )
    )
    total_chain_transitions = total_batched_transitions * active.chain_count
    maximum_work = {
        "metric_adaptation_batched_transitions": metric_transitions,
        "initial_candidate_batched_transitions": initial_candidate_transitions,
        "extension_candidate_batched_transitions": extension_candidate_transitions,
        "exact_l_tune_batched_transitions": exact_l_tune_transitions,
        "fresh_verification_batched_transitions": fresh_verification_transitions,
        "total_batched_transitions": total_batched_transitions,
        "total_chain_transitions": total_chain_transitions,
        "leapfrog_steps_upper_bound": total_chain_transitions * leapfrog_cap,
        "target_evaluation_rows_upper_bound": (
            total_chain_transitions * (leapfrog_cap + 1)
        ),
    }
    payload: dict[str, Any] = {
        "schema": _PUBLIC_MANIFEST_SCHEMA,
        "accounting_scope_id": _ACCOUNTING_SCOPE_ID,
        "whole_launch_hmc_upper_bound": False,
        "pre_scope_hmc_work_included": False,
        "policy_id": active.policy_id,
        "policy_hash": active.policy_hash,
        "algorithm_id": algorithm,
        "run_class": classification,
        "target_dimension": dimension,
        "outer_attempt_count": len(metric),
        "chain_count": active.chain_count,
        "metric_adaptation_steps": metric,
        "selection_attempts_per_outer_attempt": selection_attempts,
        "selection_attempt_count_upper_bound": selection_attempt_total,
        "candidate_count_upper_bound": active.candidate_count_upper_bound,
        "replications_per_candidate": active.replications_per_candidate,
        "candidate_replication_slots_per_selection_attempt": (
            candidate_replication_slots
        ),
        "initial_candidate_results": active.initial_candidate_results,
        "candidate_burnin_steps": active.candidate_burnin_steps,
        "evidence_extension_checkpoints": active.evidence_extension_checkpoints,
        "exact_l_tune_adaptation_steps": active.exact_l_tune_adaptation_steps,
        "exact_l_tune_result_steps": active.exact_l_tune_result_steps,
        "exact_l_tune_start_count_upper_bound": (
            selection_attempt_total * retune_starts_per_selection_attempt
        ),
        "fresh_verification_results": active.fresh_verification_results,
        "fresh_verification_burnin_steps": (
            active.fresh_verification_burnin_steps
        ),
        "fresh_verification_start_count_upper_bound": verification_start_count,
        "max_leapfrog_steps": leapfrog_cap,
        "maximum_work": maximum_work,
        "aggregate_counts_public_safe": True,
        "private_hmc_mechanics_exposed": False,
        "reports_posterior_convergence": False,
        "reports_sampler_superiority": False,
        "reports_default_readiness": False,
        "reports_gpu_or_xla_readiness": False,
        "nonclaims": OPERATIONAL_HMC_BUDGET_NONCLAIMS,
    }
    _assert_public_payload_safe(payload)
    payload["manifest_hash"] = _canonical_hash(payload)
    return payload


def build_serious_public_hmc_work_manifest(
    *,
    target_dimension: int,
    outer_attempt_count: int,
    max_leapfrog_steps: int = 25,
    policy: HMCOperationalStatisticalWorkPolicy | None = None,
) -> Mapping[str, Any]:
    attempts = _strict_positive_int(outer_attempt_count, name="outer_attempt_count")
    return build_public_hmc_work_manifest(
        target_dimension=target_dimension,
        metric_adaptation_steps=serious_metric_adaptation_schedule(
            target_dimension=target_dimension,
            outer_attempt_count=attempts,
        ),
        selection_attempts_per_outer_attempt=tuple(
            min(5, attempts - index) for index in range(attempts)
        ),
        max_leapfrog_steps=max_leapfrog_steps,
        policy=policy,
        run_class="serious",
    )


def validate_public_hmc_work_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("public work manifest must be a mapping")
    manifest = dict(payload)
    if manifest.get("schema") != _PUBLIC_MANIFEST_SCHEMA:
        raise ValueError("public work manifest schema mismatch")
    observed_hash = str(manifest.pop("manifest_hash", ""))
    _assert_public_payload_safe(manifest)
    if not observed_hash or observed_hash != _canonical_hash(manifest):
        raise ValueError("public work manifest hash mismatch")
    maximum = manifest.get("maximum_work")
    if not isinstance(maximum, Mapping):
        raise ValueError("public work manifest maximum_work is missing")
    for key, value in maximum.items():
        _strict_nonnegative_int(value, name=f"maximum_work.{key}")
    if manifest.get("aggregate_counts_public_safe") is not True:
        raise ValueError("public work manifest did not declare aggregate-count safety")
    if manifest.get("accounting_scope_id") != _ACCOUNTING_SCOPE_ID:
        raise ValueError("public work manifest accounting scope mismatch")
    if manifest.get("whole_launch_hmc_upper_bound") is not False:
        raise ValueError("public work manifest whole-launch scope is ambiguous")
    if manifest.get("pre_scope_hmc_work_included") is not False:
        raise ValueError("public work manifest pre-scope accounting is ambiguous")
    policy = HMCOperationalStatisticalWorkPolicy(
        initial_candidate_results=manifest.get("initial_candidate_results"),
        candidate_burnin_steps=manifest.get("candidate_burnin_steps"),
        evidence_extension_checkpoints=manifest.get(
            "evidence_extension_checkpoints", ()
        ),
        exact_l_tune_adaptation_steps=manifest.get(
            "exact_l_tune_adaptation_steps"
        ),
        fresh_verification_results=manifest.get("fresh_verification_results"),
        fresh_verification_burnin_steps=manifest.get(
            "fresh_verification_burnin_steps"
        ),
        candidate_count_upper_bound=manifest.get("candidate_count_upper_bound"),
        replications_per_candidate=manifest.get("replications_per_candidate"),
        exact_l_tune_result_steps=manifest.get("exact_l_tune_result_steps"),
        fresh_verification_starts_per_outer_attempt=(
            _strict_positive_int(
                manifest.get("fresh_verification_start_count_upper_bound"),
                name="fresh_verification_start_count_upper_bound",
            )
            // _strict_positive_int(
                manifest.get("outer_attempt_count"),
                name="outer_attempt_count",
            )
        ),
        chain_count=manifest.get("chain_count"),
        policy_id=str(manifest.get("policy_id", "")),
    )
    expected = build_public_hmc_work_manifest(
        target_dimension=manifest.get("target_dimension"),
        metric_adaptation_steps=manifest.get("metric_adaptation_steps", ()),
        selection_attempts_per_outer_attempt=manifest.get(
            "selection_attempts_per_outer_attempt", ()
        ),
        max_leapfrog_steps=manifest.get("max_leapfrog_steps"),
        policy=policy,
        algorithm_id=str(manifest.get("algorithm_id", "")),
        run_class=str(manifest.get("run_class", "")),
    )
    restored = {**manifest, "manifest_hash": observed_hash}
    if _canonical_hash(restored) != _canonical_hash(expected):
        raise ValueError("public work manifest arithmetic or policy mismatch")
    return restored


def build_private_resolved_hmc_work_manifest(
    *,
    public_manifest: Mapping[str, Any],
    resolved_candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    public = validate_public_hmc_work_manifest(public_manifest)
    candidates = tuple(dict(item) for item in resolved_candidates)
    payload: dict[str, Any] = {
        "schema": _PRIVATE_MANIFEST_SCHEMA,
        "public_manifest_hash": public["manifest_hash"],
        "resolved_candidate_count": len(candidates),
        "resolved_candidates": candidates,
        "private_handoff_only": True,
        "publicized": False,
    }
    payload["private_manifest_hash"] = _canonical_hash(payload)
    return payload


def validate_private_resolved_hmc_work_manifest(
    payload: Mapping[str, Any],
    *,
    expected_public_manifest_hash: str | None = None,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("private work manifest must be a mapping")
    manifest = dict(payload)
    if manifest.get("schema") != _PRIVATE_MANIFEST_SCHEMA:
        raise ValueError("private work manifest schema mismatch")
    observed_hash = str(manifest.pop("private_manifest_hash", ""))
    if not observed_hash or observed_hash != _canonical_hash(manifest):
        raise ValueError("private work manifest hash mismatch")
    public_hash = str(manifest.get("public_manifest_hash", ""))
    if not public_hash:
        raise ValueError("private work manifest public link is missing")
    if (
        expected_public_manifest_hash is not None
        and public_hash != str(expected_public_manifest_hash)
    ):
        raise ValueError("private work manifest public link mismatch")
    candidates = manifest.get("resolved_candidates")
    if not isinstance(candidates, (list, tuple)):
        raise ValueError("private work manifest candidates are missing")
    if int(manifest.get("resolved_candidate_count", -1)) != len(candidates):
        raise ValueError("private work manifest candidate count mismatch")
    if manifest.get("private_handoff_only") is not True:
        raise ValueError("private work manifest lost private-handoff status")
    if manifest.get("publicized") is not False:
        raise ValueError("private work manifest was marked public")
    return {**manifest, "private_manifest_hash": observed_hash}


def reconcile_executed_hmc_work(
    *,
    public_manifest: Mapping[str, Any],
    executed_work: Mapping[str, int],
) -> Mapping[str, Any]:
    public = validate_public_hmc_work_manifest(public_manifest)
    maximum = dict(public["maximum_work"])
    executed: dict[str, int] = {}
    for key, raw_value in executed_work.items():
        if key not in maximum:
            raise ValueError(f"executed work key has no public bound: {key}")
        executed[key] = _strict_nonnegative_int(
            raw_value,
            name=f"executed_work.{key}",
        )
    component_keys = (
        "metric_adaptation_batched_transitions",
        "initial_candidate_batched_transitions",
        "extension_candidate_batched_transitions",
        "exact_l_tune_batched_transitions",
        "fresh_verification_batched_transitions",
    )
    if "total_batched_transitions" in executed:
        missing = tuple(key for key in component_keys if key not in executed)
        if missing:
            raise ValueError(
                "executed total requires every component count: " + ", ".join(missing)
            )
        if executed["total_batched_transitions"] != sum(
            executed[key] for key in component_keys
        ):
            raise ValueError("executed total does not equal its component counts")
    exceeded = tuple(
        sorted(key for key, value in executed.items() if value > int(maximum[key]))
    )
    payload: dict[str, Any] = {
        "schema": _RECONCILIATION_SCHEMA,
        "public_manifest_hash": public["manifest_hash"],
        "executed_work": executed,
        "bound_exceeded_keys": exceeded,
        "within_public_bounds": not exceeded,
        "aggregate_counts_public_safe": True,
        "private_hmc_mechanics_exposed": False,
        "reports_posterior_convergence": False,
        "reports_sampler_superiority": False,
        "reports_default_readiness": False,
        "reports_gpu_or_xla_readiness": False,
    }
    payload["reconciliation_hash"] = _canonical_hash(payload)
    if exceeded:
        raise ValueError(
            "executed HMC work exceeded public bounds: " + ", ".join(exceeded)
        )
    return payload


def validate_executed_hmc_work_reconciliation(
    payload: Mapping[str, Any],
    *,
    expected_public_manifest_hash: str | None = None,
    public_manifest: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("work reconciliation must be a mapping")
    reconciliation = dict(payload)
    if reconciliation.get("schema") != _RECONCILIATION_SCHEMA:
        raise ValueError("work reconciliation schema mismatch")
    observed_hash = str(reconciliation.pop("reconciliation_hash", ""))
    if not observed_hash or observed_hash != _canonical_hash(reconciliation):
        raise ValueError("work reconciliation hash mismatch")
    public_hash = str(reconciliation.get("public_manifest_hash", ""))
    if not public_hash:
        raise ValueError("work reconciliation public link is missing")
    if (
        expected_public_manifest_hash is not None
        and public_hash != str(expected_public_manifest_hash)
    ):
        raise ValueError("work reconciliation public link mismatch")
    executed = reconciliation.get("executed_work")
    if not isinstance(executed, Mapping):
        raise ValueError("work reconciliation executed counts are missing")
    for key, value in executed.items():
        _strict_nonnegative_int(value, name=f"executed_work.{key}")
    if tuple(reconciliation.get("bound_exceeded_keys", ())) or (
        reconciliation.get("within_public_bounds") is not True
    ):
        raise ValueError("work reconciliation records an exceeded bound")
    restored = {**reconciliation, "reconciliation_hash": observed_hash}
    if public_manifest is None:
        raise ValueError(
            "public manifest is required to validate reconciliation arithmetic"
        )
    public = validate_public_hmc_work_manifest(public_manifest)
    if public["manifest_hash"] != public_hash:
        raise ValueError("work reconciliation public manifest mismatch")
    expected = reconcile_executed_hmc_work(
        public_manifest=public,
        executed_work=executed,
    )
    if _canonical_hash(restored) != _canonical_hash(expected):
        raise ValueError("work reconciliation arithmetic mismatch")
    return restored
