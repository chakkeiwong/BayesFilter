"""Replay-gated continuation of operational Phase-5 HMC evidence.

This module extends an in-memory, deterministically replayed Phase-5 selection.
It does not deserialize raw HMC state from aggregate artifacts: the caller must
first reproduce the source run, and the archived aggregate ledger must match
exactly before any new transition is allowed.  BayesFilter remains responsible
for every seed, HMC transition, exact-L retune, and fresh verification.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping

from bayesfilter.hmc_budget_contract import (
    HMCOperationalStatisticalWorkPolicy,
    build_private_resolved_hmc_work_manifest,
    build_public_hmc_work_manifest,
    reconcile_executed_hmc_work,
)
from bayesfilter.inference.hmc import run_full_chain_tfp_hmc, stable_adapter_signature
from bayesfilter.inference.hmc_kernel_selection import (
    BoundedFixedTrajectorySelectionResult,
    FixedTrajectoryCandidateHandoffLineage,
    FixedTrajectorySelection,
    _finalize_operational_selection_nomination,
    extend_operational_fixed_trajectory_evidence,
    select_fixed_trajectory_representative,
)
from bayesfilter.inference.hmc_kernel_tuning import (
    HMCFixedMassStepStageResult,
    HMCKernelTuningResult,
    _build_fixed_mass_hmc_adapter,
    _default_attempt_budget_policy,
    _mass_artifact_signature,
    _phase4_adapted_mass_artifact,
    _phase4_latent_adapter_for_step_stage,
    _phase7_verification_initial_state,
    _public_final_kernel_handoff_payload,
    _public_loop_config,
    run_hmc_tune_verify_repair_loop,
)
from bayesfilter.inference.hmc_verification import HMCAcceptancePolicy
from bayesfilter.runtime import stable_config_hash


_MIXED_HANDOFF_POLICY = "mixed_evidence_requires_fresh_verification"
_RESUME_SCHEMA = "bayesfilter.hmc_phase5_replay_gated_evidence_extension.v1"


@dataclass(frozen=True)
class Phase5EvidenceReplayContract:
    """Deterministic archive fields that must match before extension work."""

    evidence_ledger: Mapping[str, Any]
    adapted_mass_artifact_signature: str
    coordinate_signature: str
    metric_signature: str
    active_start_bank_signature: str
    source_start_bank_signature: str
    fixed_hmc_adapter_signature: str
    initial_step_size: float
    selection_root_seed: tuple[int, int]
    source_checkpoint: int
    extension_checkpoint: int
    target_scope: str
    verification_num_results: int
    verification_num_burnin_steps: int
    verification_start_count: int

    def __post_init__(self) -> None:
        ledger = dict(self.evidence_ledger)
        if ledger.get("schema") != (
            "bayesfilter.hmc_fixed_trajectory_private_evidence_ledger.v1"
        ):
            raise ValueError("archived Phase-5 evidence ledger schema mismatch")
        for name in (
            "adapted_mass_artifact_signature",
            "coordinate_signature",
            "metric_signature",
            "active_start_bank_signature",
            "source_start_bank_signature",
            "fixed_hmc_adapter_signature",
            "target_scope",
        ):
            value = str(getattr(self, name))
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        seed = tuple(int(item) for item in self.selection_root_seed)
        if len(seed) != 2 or seed == (0, 0):
            raise ValueError("selection_root_seed must be a nonzero seed pair")
        source = int(self.source_checkpoint)
        extension = int(self.extension_checkpoint)
        if source <= 0 or extension != 2 * source:
            raise ValueError("extension checkpoint must be the next exact doubling")
        step = float(self.initial_step_size)
        if not step > 0.0:
            raise ValueError("initial_step_size must be positive")
        verification_results = int(self.verification_num_results)
        verification_burnin = int(self.verification_num_burnin_steps)
        verification_starts = int(self.verification_start_count)
        if verification_results <= 0 or verification_burnin <= 0:
            raise ValueError("verification result and burn-in budgets must be positive")
        if verification_starts not in {2, 3}:
            raise ValueError("verification_start_count must be 2 or 3")
        attempts = tuple(ledger.get("attempts", ()))
        if len(attempts) != 1:
            raise ValueError("resume accepts exactly one archived selection attempt")
        extensions = tuple(attempts[0].get("extensions", ()))
        if tuple(item.get("checkpoint") for item in extensions) != (source,):
            raise ValueError("archive does not terminate at the declared checkpoint")
        object.__setattr__(self, "evidence_ledger", ledger)
        object.__setattr__(self, "selection_root_seed", seed)
        object.__setattr__(self, "source_checkpoint", source)
        object.__setattr__(self, "extension_checkpoint", extension)
        object.__setattr__(self, "initial_step_size", step)
        object.__setattr__(self, "verification_num_results", verification_results)
        object.__setattr__(
            self,
            "verification_num_burnin_steps",
            verification_burnin,
        )
        object.__setattr__(self, "verification_start_count", verification_starts)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_phase5_evidence_replay_contract.v1",
            "archived_bounded_selection_signature": self.evidence_ledger.get(
                "bounded_selection_signature"
            ),
            "adapted_mass_artifact_signature": (
                self.adapted_mass_artifact_signature
            ),
            "coordinate_signature": self.coordinate_signature,
            "metric_signature": self.metric_signature,
            "active_start_bank_signature": self.active_start_bank_signature,
            "source_start_bank_signature": self.source_start_bank_signature,
            "fixed_hmc_adapter_signature": self.fixed_hmc_adapter_signature,
            "initial_step_size": self.initial_step_size,
            "selection_root_seed": self.selection_root_seed,
            "source_checkpoint": self.source_checkpoint,
            "extension_checkpoint": self.extension_checkpoint,
            "target_scope": self.target_scope,
            "verification_num_results": self.verification_num_results,
            "verification_num_burnin_steps": self.verification_num_burnin_steps,
            "verification_start_count": self.verification_start_count,
            "runtime_bearing_artifact_hashes_are_replay_vetoes": False,
            "private_handoff_only": True,
        }


@dataclass(frozen=True)
class Phase5EvidenceExtensionResult:
    """Typed result of one replay-gated evidence continuation."""

    source_tuning_result: HMCKernelTuningResult
    tuning_result: HMCKernelTuningResult
    replay_contract: Phase5EvidenceReplayContract
    replay_gate: Mapping[str, Any]
    extension_payload: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if not isinstance(self.source_tuning_result, HMCKernelTuningResult):
            raise TypeError("source_tuning_result has invalid type")
        if not isinstance(self.tuning_result, HMCKernelTuningResult):
            raise TypeError("tuning_result has invalid type")
        if not isinstance(self.replay_contract, Phase5EvidenceReplayContract):
            raise TypeError("replay_contract has invalid type")
        gate = dict(self.replay_gate)
        if gate.get("passed") is not True:
            raise ValueError("typed continuation result requires a passed replay gate")
        object.__setattr__(self, "replay_gate", gate)
        object.__setattr__(
            self,
            "extension_payload",
            None if self.extension_payload is None else dict(self.extension_payload),
        )

    @property
    def passed(self) -> bool:
        return self.tuning_result.passed

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": _RESUME_SCHEMA,
            "source_tuning_artifact_hash": self.source_tuning_result.artifact_hash,
            "continuation_tuning_artifact_hash": self.tuning_result.artifact_hash,
            "replay_contract": self.replay_contract.payload(),
            "replay_gate": self.replay_gate,
            "extension": self.extension_payload,
            "final_status": self.tuning_result.final_status,
            "diagnostic_role": self.tuning_result.diagnostic_role,
            "hard_vetoes": self.tuning_result.hard_vetoes,
            "repair_triggers": self.tuning_result.repair_triggers,
            "retained_sampling_authorized": False,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "private_handoff_only": True,
        }


def _source_fixed_stage(
    result: HMCKernelTuningResult,
) -> tuple[Any, HMCFixedMassStepStageResult]:
    loop = result.tune_verify_repair_loop
    if loop is None or len(loop.attempts) != 1:
        raise ValueError("source replay must contain exactly one Phase-7 attempt")
    attempt = loop.attempts[0]
    if attempt.windowed_stage is None or attempt.fixed_mass_step_stage is None:
        raise ValueError("source replay lacks its in-memory Phase-4/Phase-5 handoff")
    fixed = attempt.fixed_mass_step_stage
    if fixed._operational_selection_loop is None:
        raise ValueError("source replay lacks typed operational selection evidence")
    return attempt.windowed_stage, fixed


def validate_phase5_evidence_replay(
    *,
    source_tuning_result: HMCKernelTuningResult,
    contract: Phase5EvidenceReplayContract,
) -> Mapping[str, Any]:
    """Prove deterministic source equality without comparing elapsed time."""

    windowed, fixed = _source_fixed_stage(source_tuning_result)
    operational = windowed.operational_warmup_result
    if operational is None:
        raise ValueError("source replay did not use operational warmup")
    loop = fixed._operational_selection_loop
    assert loop is not None
    observed_ledger = loop.private_evidence_ledger()
    checks = {
        "aggregate_evidence_ledger": (
            stable_config_hash(observed_ledger)
            == stable_config_hash(contract.evidence_ledger)
        ),
        "bounded_selection_signature": (
            loop.signature
            == contract.evidence_ledger.get("bounded_selection_signature")
        ),
        "adapted_mass_artifact_signature": (
            fixed.adapted_mass_artifact_signature
            == contract.adapted_mass_artifact_signature
        ),
        "coordinate_signature": (
            operational.final_kernel_state.transform.signature
            == contract.coordinate_signature
        ),
        "metric_signature": (
            operational.final_kernel_state.momentum_metric.signature
            == contract.metric_signature
        ),
        "active_start_bank_signature": (
            fixed.diagnostics.get("frozen_start_bank_signature")
            == contract.active_start_bank_signature
        ),
        "source_start_bank_signature": (
            operational.private_start_bank_signature
            == contract.source_start_bank_signature
        ),
        "fixed_hmc_adapter_signature": (
            fixed.ladder_hmc_adapter_signature
            == contract.fixed_hmc_adapter_signature
        ),
        "initial_step_size": fixed.initial_step_size == contract.initial_step_size,
        "selection_root_seed": (
            tuple(fixed.config.seed) == contract.selection_root_seed
        ),
        "target_scope": fixed.config.target_scope == contract.target_scope,
        "source_checkpoint": (
            loop.evidence_extension_checkpoints == (contract.source_checkpoint,)
        ),
        "source_terminal_disposition": (
            loop.terminal_disposition == "budget_exhausted_valid"
            and loop.selection.disposition == "inconclusive_evidence"
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "Phase-5 deterministic replay mismatch before extension: "
            + ", ".join(failed)
        )
    return {
        "schema": "bayesfilter.hmc_phase5_evidence_replay_gate.v1",
        "passed": True,
        "checks": checks,
        "archived_bounded_selection_signature": contract.evidence_ledger[
            "bounded_selection_signature"
        ],
        "observed_bounded_selection_signature": loop.signature,
        "runtime_bearing_artifact_hashes_compared": False,
        "extension_transition_count_before_gate": 0,
        "private_handoff_only": True,
    }


def _extended_selection_loop(
    *,
    source_loop: BoundedFixedTrajectorySelectionResult,
    finalized: FixedTrajectorySelection,
    extension: Any,
    adapter: Any,
    bank: Any,
    bank_signature: str,
    coordinate_signature: str,
    metric_signature: str,
    frozen_step_size: float,
    root_seed: tuple[int, int],
    target_scope: str,
    acceptance_policy: HMCAcceptancePolicy,
    final_tune_adaptation_steps: int,
    chain_execution_mode: str,
    use_xla: bool,
    target_status_trace_policy: str,
) -> BoundedFixedTrajectorySelectionResult:
    source_attempt = source_loop.attempts[0]
    selection = finalized
    handoff = None
    if selection.disposition == "inconclusive_evidence":
        strict_source = selection
        provisional = select_fixed_trajectory_representative(
            strict_source.candidate_results,
            anchor_l=strict_source.anchor_l,
            candidate_handoff_policy=_MIXED_HANDOFF_POLICY,
            terminal_candidate_handoff=True,
        )
        if provisional.disposition == "representative_nominated":
            selection = _finalize_operational_selection_nomination(
                selection=provisional,
                adapter=adapter,
                bank=bank,
                bank_signature=bank_signature,
                coordinate_signature=coordinate_signature,
                metric_signature=metric_signature,
                frozen_step_size=frozen_step_size,
                root_seed=root_seed,
                target_scope=target_scope,
                acceptance_policy=acceptance_policy,
                final_tune_adaptation_steps=final_tune_adaptation_steps,
                chain_execution_mode=chain_execution_mode,
                use_xla=use_xla,
                target_status_trace_policy=target_status_trace_policy,
                runner=run_full_chain_tfp_hmc,
            )
            disposition = "mixed_evidence_provisional_nomination"
        else:
            selection = strict_source
            provisional = None
            disposition = "mixed_evidence_rejected"
        handoff = FixedTrajectoryCandidateHandoffLineage(
            policy=_MIXED_HANDOFF_POLICY,
            checkpoint=extension.checkpoint,
            source_selection=strict_source,
            provisional_selection=provisional,
            finalized_selection=selection,
            disposition=disposition,
        )
    attempt = dataclasses.replace(
        source_attempt,
        selection=selection,
        evidence_extensions=(*source_attempt.evidence_extensions, extension),
        candidate_handoff_lineage=handoff,
    )
    terminal = {
        "representative_selected": "representative_selected",
        "inconclusive_evidence": "budget_exhausted_valid",
    }.get(selection.disposition, selection.disposition)
    return BoundedFixedTrajectorySelectionResult(
        attempts=(attempt,),
        max_attempts=1,
        terminal_disposition=terminal,
        evidence_extension_checkpoints=(
            source_loop.evidence_extension_checkpoints[0],
            extension.checkpoint,
        ),
        candidate_handoff_policy=_MIXED_HANDOFF_POLICY,
    )


def _extend_fixed_stage(
    *,
    adapter: Any,
    geometry: Any,
    windowed_stage: Any,
    source_fixed_stage: HMCFixedMassStepStageResult,
    contract: Phase5EvidenceReplayContract,
) -> tuple[HMCFixedMassStepStageResult, Mapping[str, Any]]:
    source_loop = source_fixed_stage._operational_selection_loop
    if source_loop is None:
        raise ValueError("source fixed stage lacks its typed selection loop")
    operational = windowed_stage.operational_warmup_result
    if operational is None:
        raise ValueError("source fixed stage lacks operational warmup")
    adapted_mass = _phase4_adapted_mass_artifact(windowed_stage)
    mass_signature = _mass_artifact_signature(adapted_mass)
    phase4_adapter = _phase4_latent_adapter_for_step_stage(
        adapter=adapter,
        geometry=geometry,
        windowed_stage=windowed_stage,
        target_scope=contract.target_scope,
    )
    final_adapter = _build_fixed_mass_hmc_adapter(
        adapter=phase4_adapter,
        mass_artifact=adapted_mass,
        mass_signature=mass_signature,
        target_scope=contract.target_scope,
    )
    final_signature = stable_adapter_signature(final_adapter)
    if final_signature != contract.fixed_hmc_adapter_signature:
        raise ValueError("live fixed-mass adapter signature changed before extension")
    bank, start_summary = _phase7_verification_initial_state(
        windowed_stage=windowed_stage,
        phase4_adapter=phase4_adapter,
        verification_adapter=final_adapter,
        verification_hmc_signature=final_signature,
    )
    bank_signature = str(start_summary.get("active_signature", ""))
    if bank_signature != contract.active_start_bank_signature:
        raise ValueError("live start-bank signature changed before extension")
    policy = HMCAcceptancePolicy(
        target=source_fixed_stage.config.target_accept_prob,
        practical_region=source_fixed_stage.config.acceptance_band,
        repair_region=source_fixed_stage.config.repair_band,
    )
    finalized, extension = extend_operational_fixed_trajectory_evidence(
        selection=source_loop.selection,
        adapter=final_adapter,
        private_start_bank=bank,
        private_start_bank_signature=bank_signature,
        coordinate_signature=contract.coordinate_signature,
        metric_signature=contract.metric_signature,
        frozen_step_size=contract.initial_step_size,
        root_seed=contract.selection_root_seed,
        target_scope=contract.target_scope,
        acceptance_policy=policy,
        checkpoint=contract.extension_checkpoint,
        extension_round_index=1,
        screen_num_burnin_steps=int(
            source_fixed_stage.diagnostics["selection_burnin_steps"]
        ),
        final_tune_adaptation_steps=int(
            source_fixed_stage.diagnostics["exact_l_tune_adaptation_steps"]
        ),
        chain_execution_mode=source_fixed_stage.config.chain_execution_mode,
        use_xla=source_fixed_stage.config.use_xla,
        target_status_trace_policy=(
            source_fixed_stage.config.target_status_trace_policy
        ),
        run_full_chain=run_full_chain_tfp_hmc,
    )
    loop = _extended_selection_loop(
        source_loop=source_loop,
        finalized=finalized,
        extension=extension,
        adapter=final_adapter,
        bank=bank,
        bank_signature=bank_signature,
        coordinate_signature=contract.coordinate_signature,
        metric_signature=contract.metric_signature,
        frozen_step_size=contract.initial_step_size,
        root_seed=contract.selection_root_seed,
        target_scope=contract.target_scope,
        acceptance_policy=policy,
        final_tune_adaptation_steps=int(
            source_fixed_stage.diagnostics["exact_l_tune_adaptation_steps"]
        ),
        chain_execution_mode=source_fixed_stage.config.chain_execution_mode,
        use_xla=source_fixed_stage.config.use_xla,
        target_status_trace_policy=(
            source_fixed_stage.config.target_status_trace_policy
        ),
    )
    selection = loop.selection
    representative = selection.representative
    selected_payload = None
    selected_hash = None
    if selection.disposition == "representative_selected":
        if representative is None or representative.exact_l_retuned_step_size is None:
            raise ValueError("extended selection lost its exact-L representative")
        selected_payload = {
            "schema": "bayesfilter.hmc_operational_exact_l_step.v2",
            "step_size": representative.exact_l_retuned_step_size,
            "num_leapfrog_steps": representative.candidate.num_leapfrog_steps,
            "selection_signature": selection.signature,
            "candidate_signature": representative.candidate.signature,
            "exact_l_retune_signature": representative.exact_l_retune_signature,
            "coordinate_signature": representative.candidate.coordinate_signature,
            "metric_signature": representative.candidate.metric_signature,
            "start_bank_signature": representative.candidate.start_bank_signature,
            "private_handoff_only": True,
        }
        selected_hash = stable_config_hash(selected_payload)

    work_policy = HMCOperationalStatisticalWorkPolicy(
        initial_candidate_results=int(
            source_fixed_stage.diagnostics["selection_initial_results"]
        ),
        candidate_burnin_steps=int(
            source_fixed_stage.diagnostics["selection_burnin_steps"]
        ),
        evidence_extension_checkpoints=(
            contract.source_checkpoint,
            contract.extension_checkpoint,
        ),
        exact_l_tune_adaptation_steps=int(
            source_fixed_stage.diagnostics["exact_l_tune_adaptation_steps"]
        ),
        fresh_verification_results=contract.verification_num_results,
        fresh_verification_burnin_steps=contract.verification_num_burnin_steps,
        fresh_verification_starts_per_outer_attempt=(
            contract.verification_start_count
        ),
    )
    public_manifest = build_public_hmc_work_manifest(
        target_dimension=windowed_stage.target_dimension,
        metric_adaptation_steps=(operational.config.warmup_steps,),
        selection_attempts_per_outer_attempt=(1,),
        max_leapfrog_steps=max(
            item.candidate.max_leapfrog_steps
            for item in selection.candidate_results
        ),
        policy=work_policy,
        run_class="phase5_replay_gated_evidence_extension",
    )
    resolved_candidates = tuple(
        {
            "selection_attempt_index": attempt.attempt_index,
            "candidate": candidate.candidate.payload(),
        }
        for attempt in loop.attempts
        for candidate in attempt.selection.candidate_results
    )
    private_manifest = build_private_resolved_hmc_work_manifest(
        public_manifest=public_manifest,
        resolved_candidates=resolved_candidates,
    )
    attempt = loop.attempts[0]
    initial_work = len(attempt.initial_replication_seeds) * (
        work_policy.initial_candidate_results + work_policy.candidate_burnin_steps
    )
    extension_work = sum(
        len(item.slots) * (item.checkpoint + work_policy.candidate_burnin_steps)
        for item in attempt.evidence_extensions
    )
    retune_work = len(attempt.exact_l_retune_seeds) * (
        work_policy.exact_l_tune_adaptation_steps
        + work_policy.exact_l_tune_result_steps
    )
    reconciliation = reconcile_executed_hmc_work(
        public_manifest=public_manifest,
        executed_work={
            "initial_candidate_batched_transitions": initial_work,
            "extension_candidate_batched_transitions": extension_work,
            "exact_l_tune_batched_transitions": retune_work,
        },
    )
    if selection.disposition == "representative_selected":
        final_status = "passed"
        diagnostic_role = "operational_fixed_trajectory_handoff_only"
        hard_vetoes: tuple[str, ...] = ()
        repair_triggers: tuple[str, ...] = ()
    elif selection.disposition == "shared_invalidity":
        final_status = "hard_veto"
        diagnostic_role = "shared_invalidity"
        hard_vetoes = ("operational_candidate_shared_invalidity",)
        repair_triggers = ()
    elif selection.disposition == "candidate_retune_failed":
        final_status = "budget_exhausted"
        diagnostic_role = "candidate_retune_failed_non_promoting"
        hard_vetoes = ()
        repair_triggers = ("operational_candidate_retune_failed",)
    else:
        final_status = "budget_exhausted"
        diagnostic_role = loop.terminal_disposition
        hard_vetoes = ()
        repair_triggers = (
            f"operational_candidate_{loop.terminal_disposition}",
        )
    diagnostics = {
        **dict(source_fixed_stage.diagnostics),
        "passed": final_status == "passed",
        "selection_disposition": selection.disposition,
        "selection_signature": selection.signature,
        "selection_loop_signature": loop.signature,
        "selection_attempt_count": 1,
        "selection_attempt_budget": 1,
        "selection_evidence_extension_checkpoints": (
            contract.source_checkpoint,
            contract.extension_checkpoint,
        ),
        "selection_terminal_disposition": loop.terminal_disposition,
        "selection_retune_failure_scope": selection.retune_failure_scope,
        "selection_retune_failure_reasons": selection.retune_failure_reasons,
        "selection_retune_candidate_signature": selection.retune_candidate_signature,
        "selection_candidate_retune_failures": tuple(
            item.payload() for item in selection.candidate_retune_failures
        ),
        "selection_candidate_retune_failure_count": len(
            selection.candidate_retune_failures
        ),
        "representative_signature": selection.representative_signature,
        "selected_num_leapfrog_steps": None
        if representative is None
        else representative.candidate.num_leapfrog_steps,
        "selected_step_size": None
        if representative is None
        else representative.exact_l_retuned_step_size,
        "operational_budget_policy_hash": work_policy.policy_hash,
        "public_work_manifest_hash": public_manifest["manifest_hash"],
        "private_work_manifest_hash": private_manifest["private_manifest_hash"],
        "executed_work_reconciliation": reconciliation,
        "phase5_replay_gate_passed_before_extension": True,
        "phase5_extension_checkpoint": contract.extension_checkpoint,
        "phase5_extension_slot_count": len(extension.slots),
    }
    fixed = dataclasses.replace(
        source_fixed_stage,
        fixed_num_leapfrog_steps=(
            source_fixed_stage.fixed_num_leapfrog_steps
            if representative is None
            else representative.candidate.num_leapfrog_steps
        ),
        final_status=final_status,
        diagnostic_role=diagnostic_role,
        hard_vetoes=hard_vetoes,
        repair_triggers=repair_triggers,
        diagnostics=diagnostics,
        selected_step_payload=selected_payload,
        selected_step_hash=selected_hash,
        repair_step_payload=None,
        repair_step_hash=None,
        _operational_selection=selection,
        _operational_selection_loop=loop,
        _operational_private_work_manifest=private_manifest,
        _operational_public_work_manifest=public_manifest,
    )
    return fixed, extension.payload()


def run_replay_gated_phase5_evidence_extension(
    *,
    adapter: Any,
    source_tuning_result: HMCKernelTuningResult,
    contract: Phase5EvidenceReplayContract,
) -> Phase5EvidenceExtensionResult:
    """Extend checkpoint evidence only after exact deterministic replay."""

    if not isinstance(source_tuning_result, HMCKernelTuningResult):
        raise TypeError("source_tuning_result has invalid type")
    if not isinstance(contract, Phase5EvidenceReplayContract):
        raise TypeError("contract has invalid type")
    if stable_adapter_signature(adapter) != source_tuning_result.adapter_signature:
        raise ValueError("live adapter does not match the source replay")
    if source_tuning_result.geometry is None or source_tuning_result.bootstrap is None:
        raise ValueError("source replay lacks geometry or bootstrap state")
    replay_gate = validate_phase5_evidence_replay(
        source_tuning_result=source_tuning_result,
        contract=contract,
    )
    windowed, source_fixed = _source_fixed_stage(source_tuning_result)
    loop_config = _public_loop_config(source_tuning_result.config)
    verification_starts = contract.verification_start_count
    extension_payload: dict[str, Any] = {}
    extended_stage: HMCFixedMassStepStageResult | None = None
    windowed_call_count = 0
    fixed_call_count = 0

    def windowed_runner(**_kwargs: Any) -> Any:
        nonlocal windowed_call_count
        windowed_call_count += 1
        if windowed_call_count != 1:
            raise ValueError("Phase-5 continuation cannot restart mass adaptation")
        return windowed

    def fixed_runner(**_kwargs: Any) -> HMCFixedMassStepStageResult:
        nonlocal extended_stage, fixed_call_count
        fixed_call_count += 1
        if fixed_call_count != 1:
            raise ValueError("Phase-5 continuation cannot restart fixed selection")
        fixed, extension = _extend_fixed_stage(
            adapter=adapter,
            geometry=source_tuning_result.geometry,
            windowed_stage=windowed,
            source_fixed_stage=source_fixed,
            contract=contract,
        )
        extension_payload.update(extension)
        extended_stage = fixed
        return extended_stage

    work_policy = HMCOperationalStatisticalWorkPolicy(
        initial_candidate_results=int(
            source_fixed.diagnostics["selection_initial_results"]
        ),
        candidate_burnin_steps=int(
            source_fixed.diagnostics["selection_burnin_steps"]
        ),
        evidence_extension_checkpoints=(
            contract.source_checkpoint,
            contract.extension_checkpoint,
        ),
        exact_l_tune_adaptation_steps=int(
            source_fixed.diagnostics["exact_l_tune_adaptation_steps"]
        ),
        fresh_verification_results=contract.verification_num_results,
        fresh_verification_burnin_steps=contract.verification_num_burnin_steps,
        fresh_verification_starts_per_outer_attempt=verification_starts,
    )

    def budget_factory(target_dimension: int, attempt_index: int) -> Any:
        return _default_attempt_budget_policy(
            target_dimension,
            attempt_index,
            mass_artifact=source_tuning_result.geometry.mass_artifact,
            operational_policy=work_policy,
        )

    continuation_loop = run_hmc_tune_verify_repair_loop(
        adapter=adapter,
        geometry=source_tuning_result.geometry,
        bootstrap=source_tuning_result.bootstrap,
        config=loop_config,
        _budget_policy_factory=budget_factory,
        _windowed_stage_runner=windowed_runner,
        _fixed_mass_step_stage_runner=fixed_runner,
    )
    final_payload = (
        _public_final_kernel_handoff_payload(continuation_loop)
        if continuation_loop.passed
        else None
    )
    tuning_result = dataclasses.replace(
        source_tuning_result,
        tune_verify_repair_loop=continuation_loop,
        final_status=continuation_loop.final_status,
        diagnostic_role=continuation_loop.diagnostic_role,
        hard_vetoes=continuation_loop.hard_vetoes,
        repair_triggers=continuation_loop.repair_triggers,
        final_kernel_payload=final_payload,
        final_kernel_hash=None
        if final_payload is None
        else stable_config_hash(final_payload),
        artifact_path=None,
    )
    return Phase5EvidenceExtensionResult(
        source_tuning_result=source_tuning_result,
        tuning_result=tuning_result,
        replay_contract=contract,
        replay_gate=replay_gate,
        extension_payload=extension_payload or None,
    )
