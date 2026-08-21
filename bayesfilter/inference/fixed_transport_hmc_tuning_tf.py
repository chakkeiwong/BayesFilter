"""TensorFlow/TFP tuning for HMC behind a frozen nonlinear transport.

The public tuner owns the trajectory grid and tunes one scalar step size over
the complete rank-2 chain bank. It intentionally has no NumPy, NUTS, or mass
adaptation dependency. Tuning transitions are discarded. The compatibility
policy verifies each candidate before selection; the replicated-efficiency
policy instead ranks all ladder nominees and verifies only the selected kernel
on one fresh held-out seed before it can be handed to sequential HMC.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
    FixedTransportFullChainConfig as _FullChainHMCConfig,
    FixedTransportHMCPolicy as _TuningPolicy,
    FixedTransportReusableRunnerPool,
    RunFullChainFn,
    build_fixed_transport_value_score_adapter,
    fixed_transport_target_status_diagnostics as _target_status_diagnostics,
    fixed_transport_tensor_diagnostics as _tensor_diagnostics,
    run_fixed_transport_full_chain_tfp_hmc as _run_full_chain_tfp_hmc,
)
from bayesfilter.inference.hmc_convergence import (
    RANK_NORMALIZED_SPLIT_RHAT_DEFINITION,
    RankNormalizedHMCThresholds,
    rank_normalized_hmc_diagnostics,
    rank_normalized_split_rhat_summary,
)
from bayesfilter.inference.posterior_adapter import value_score_capability
from bayesfilter.inference.tuning_contract import (
    HMCTuningScope,
    require_active_hmc_tuning_route,
)


FIXED_TRANSPORT_HMC_TUNING_NONCLAIMS: tuple[str, ...] = (
    "fixed trained transport HMC tuning only",
    "transport is not trained or adapted by this tuner",
    "identity z-mass policy only",
    "no windowed mass adaptation claim",
    "no posterior convergence claim",
    "no sampler superiority claim",
    "no default-readiness claim",
    "no external-client scientific claim",
    "native divergence unavailability is not zero divergences",
)

_FORBIDDEN_BASE_AUTHORITIES = frozenset({"gradient_tape_fallback"})
_SELECTION_POLICIES = frozenset(
    {
        "acceptance_target_distance",
        "replicated_min_bulk_ess_per_gradient",
    }
)


@dataclass(frozen=True)
class FixedTransportHMCKernelTuningConfig:
    """Policy for fixed-NeuTra HMC kernel tuning in transport coordinates."""

    initial_step_size: float
    leapfrog_grid: tuple[int, ...] = (5, 10, 15, 20, 25)
    chain_count: int = 4
    initial_state_bank: tuple[tuple[float, ...], ...] = ()
    target_accept_prob: float = 0.70
    acceptance_band: tuple[float, float] = (0.65, 0.75)
    repair_band: tuple[float, float] = (0.55, 0.85)
    maximum_absolute_energy_error: float = 1000.0
    step_repair_factor: float = 2.0
    budget_schedule: tuple[int, ...] = (8, 16, 32)
    tune_num_results: int = 8
    screen_num_results: int = 16
    screen_num_burnin_steps: int = 4
    selection_policy: str = "acceptance_target_distance"
    selection_replications: int = 1
    selection_num_results: int = 64
    selection_num_burnin_steps: int = 32
    selection_seed_base: tuple[int, int] = (20260625, 250)
    selection_acceptance_band: tuple[float, float] | None = None
    verification_num_results: int = 16
    verification_num_burnin_steps: int = 4
    require_modern_rank_normalized_verification: bool = False
    report_modern_rank_normalized_verification: bool = False
    verification_min_retained_results_per_chain: int = 1000
    verification_rhat_max: float = 1.01
    verification_coordinate_system: str = "raw_target_coordinates"
    require_all_chain_movement: bool = True
    tune_seed_base: tuple[int, int] = (20260625, 100)
    screen_seed_base: tuple[int, int] = (20260625, 200)
    verification_seed_base: tuple[int, int] = (20260625, 300)
    chain_execution_mode: str = "tf_function"
    use_xla: bool = True
    target_scope: str | None = None
    target_status_trace_policy: str = "none"
    fixed_grid_base_step_size_candidates: tuple[float, ...] = ()
    fixed_grid_scale_candidates: tuple[float, ...] = ()
    fixed_grid_num_leapfrog_steps: int | None = None
    fixed_grid_max_attempts: int = 5
    fixed_grid_fallback_acceptance_max: float = 0.85
    output_filename: str = "fixed_transport_hmc_tuning_result.json"
    source: str = "bayesfilter.inference.fixed_transport_hmc_tuning"
    proposal_dynamics_identity: str = "exact_transformed_gradient"

    def __post_init__(self) -> None:
        step = _positive_float(self.initial_step_size, name="initial_step_size")
        object.__setattr__(self, "initial_step_size", step)
        leapfrogs = tuple(dict.fromkeys(int(value) for value in self.leapfrog_grid))
        if not leapfrogs or any(value < 2 for value in leapfrogs):
            raise ValueError("leapfrog_grid must contain integers greater than or equal to 2")
        object.__setattr__(self, "leapfrog_grid", leapfrogs)
        for name in (
            "chain_count",
            "tune_num_results",
            "screen_num_results",
            "screen_num_burnin_steps",
            "selection_replications",
            "selection_num_results",
            "selection_num_burnin_steps",
            "verification_num_results",
            "verification_num_burnin_steps",
            "verification_min_retained_results_per_chain",
            "fixed_grid_max_attempts",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        bank = tuple(
            tuple(float(value) for value in row) for row in self.initial_state_bank
        )
        if bank:
            if len(bank) != self.chain_count:
                raise ValueError("initial_state_bank row count must match chain_count")
            if not bank[0] or any(len(row) != len(bank[0]) for row in bank):
                raise ValueError("initial_state_bank rows must have one common positive width")
            if any(not math.isfinite(value) for row in bank for value in row):
                raise ValueError("initial_state_bank must be finite")
        object.__setattr__(self, "initial_state_bank", bank)
        target = float(self.target_accept_prob)
        if not math.isfinite(target) or not 0.0 < target < 1.0:
            raise ValueError("target_accept_prob must be finite and in (0, 1)")
        object.__setattr__(self, "target_accept_prob", target)
        acceptance = _validate_band(self.acceptance_band, name="acceptance_band")
        repair = _validate_band(self.repair_band, name="repair_band")
        if repair[0] > acceptance[0] or repair[1] < acceptance[1]:
            raise ValueError("repair_band must contain acceptance_band")
        object.__setattr__(self, "acceptance_band", acceptance)
        object.__setattr__(self, "repair_band", repair)
        object.__setattr__(
            self,
            "maximum_absolute_energy_error",
            _positive_float(
                self.maximum_absolute_energy_error,
                name="maximum_absolute_energy_error",
            ),
        )
        repair_factor = float(self.step_repair_factor)
        if not math.isfinite(repair_factor) or repair_factor <= 1.0:
            raise ValueError("step_repair_factor must be finite and greater than 1")
        object.__setattr__(self, "step_repair_factor", repair_factor)
        budgets = tuple(int(value) for value in self.budget_schedule)
        if not budgets or any(value <= 0 for value in budgets):
            raise ValueError("budget_schedule must contain positive integers")
        object.__setattr__(self, "budget_schedule", budgets)
        selection_policy = str(self.selection_policy)
        if selection_policy not in _SELECTION_POLICIES:
            raise ValueError(
                "selection_policy must be one of: "
                + ", ".join(sorted(_SELECTION_POLICIES))
            )
        if (
            selection_policy == "replicated_min_bulk_ess_per_gradient"
            and self.selection_num_results < 4
        ):
            raise ValueError(
                "replicated efficiency selection requires at least four results per chain"
            )
        object.__setattr__(self, "selection_policy", selection_policy)
        selection_band = (
            acceptance
            if self.selection_acceptance_band is None
            else _validate_band(
                self.selection_acceptance_band,
                name="selection_acceptance_band",
            )
        )
        object.__setattr__(self, "selection_acceptance_band", selection_band)
        if self.require_all_chain_movement is not True:
            raise ValueError("fixed-transport tuning requires all-chain movement")
        for name in (
            "tune_seed_base",
            "screen_seed_base",
            "selection_seed_base",
            "verification_seed_base",
        ):
            object.__setattr__(self, name, _validate_seed(getattr(self, name)))
        mode = str(self.chain_execution_mode)
        if mode not in {"tf_function", "eager"}:
            raise ValueError("chain_execution_mode must be 'tf_function' or 'eager'")
        if self.use_xla and mode != "tf_function":
            raise ValueError("XLA full-chain HMC requires chain_execution_mode='tf_function'")
        object.__setattr__(self, "chain_execution_mode", mode)
        status_policy = str(self.target_status_trace_policy)
        if status_policy not in {"none", "per_chain_step"}:
            raise ValueError("target_status_trace_policy is invalid")
        object.__setattr__(self, "target_status_trace_policy", status_policy)
        modern_verification_requested = bool(
            self.require_modern_rank_normalized_verification
            or self.report_modern_rank_normalized_verification
        )
        if modern_verification_requested:
            if self.chain_count != 4:
                raise ValueError("modern rank-normalized verification requires exactly four chains")
            if self.verification_num_results < self.verification_min_retained_results_per_chain:
                raise ValueError(
                    "modern rank-normalized verification requires at least the configured retained results per chain"
                )
        rhat = float(self.verification_rhat_max)
        if not math.isfinite(rhat) or rhat <= 1.0:
            raise ValueError("verification_rhat_max must be finite and greater than 1")
        object.__setattr__(self, "verification_rhat_max", rhat)
        coordinate_system = str(self.verification_coordinate_system)
        if coordinate_system not in {"raw_target_coordinates", "hmc_coordinates"}:
            raise ValueError("verification_coordinate_system is invalid")
        object.__setattr__(self, "verification_coordinate_system", coordinate_system)
        base = tuple(float(value) for value in self.fixed_grid_base_step_size_candidates)
        scales = tuple(float(value) for value in self.fixed_grid_scale_candidates)
        if any(not math.isfinite(value) or value <= 0.0 for value in (*base, *scales)):
            raise ValueError("fixed-grid values must be positive and finite")
        if bool(base) != bool(scales):
            raise ValueError("fixed-grid base and scale candidates must be supplied together")
        object.__setattr__(self, "fixed_grid_base_step_size_candidates", base)
        object.__setattr__(self, "fixed_grid_scale_candidates", scales)
        if self.fixed_grid_num_leapfrog_steps is not None:
            fixed_l = int(self.fixed_grid_num_leapfrog_steps)
            if fixed_l < 2:
                raise ValueError(
                    "fixed_grid_num_leapfrog_steps must be greater than or equal to 2"
                )
            object.__setattr__(self, "fixed_grid_num_leapfrog_steps", fixed_l)
        fallback = float(self.fixed_grid_fallback_acceptance_max)
        if not math.isfinite(fallback) or not 0.0 < fallback < 1.0:
            raise ValueError("fixed_grid_fallback_acceptance_max is invalid")
        if base and fallback < acceptance[1]:
            raise ValueError(
                "fixed_grid_fallback_acceptance_max must contain the pass-band upper bound"
            )
        object.__setattr__(self, "fixed_grid_fallback_acceptance_max", fallback)

    def payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["modern_rank_normalized_verification_role"] = (
            "required_hard_gate"
            if self.require_modern_rank_normalized_verification
            else (
                "diagnostic_only_not_handoff_gate"
                if self.report_modern_rank_normalized_verification
                else "not_requested"
            )
        )
        payload["maximum_absolute_energy_error_role"] = "explanatory_alert_only"
        payload["shared_scalar_step_across_chain_bank"] = True
        payload["runtime_numerical_backend"] = "tensorflow_tfp_only"
        payload["tuning_branch"] = (
            "fixed_grid" if self.fixed_grid_base_step_size_candidates else "dual_averaging"
        )
        payload["fixed_grid_fallback_acceptance_max_role"] = (
            "fixed_grid_acceptance_classification"
            if self.fixed_grid_base_step_size_candidates
            else "not_applicable_to_dual_averaging"
        )
        return payload


@dataclass(frozen=True)
class FixedTransportHMCCandidateResult:
    candidate_index: int
    num_leapfrog_steps: int
    ladder_result: Mapping[str, Any] | None
    verification_config_payload: Mapping[str, Any] | None
    verification_diagnostics: Mapping[str, Any]
    final_status: str
    diagnostic_role: str
    fixed_kernel_step_size: float | None = None
    hard_vetoes: tuple[str, ...] = ()
    repair_triggers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.num_leapfrog_steps) < 2:
            raise ValueError("num_leapfrog_steps must be greater than or equal to 2")
        object.__setattr__(self, "candidate_index", int(self.candidate_index))
        object.__setattr__(self, "num_leapfrog_steps", int(self.num_leapfrog_steps))
        if self.fixed_kernel_step_size is None and self.ladder_result is None:
            raise ValueError("candidate requires ladder_result or fixed_kernel_step_size")
        if self.fixed_kernel_step_size is not None:
            object.__setattr__(
                self,
                "fixed_kernel_step_size",
                _positive_float(self.fixed_kernel_step_size, name="fixed_kernel_step_size"),
            )
        object.__setattr__(self, "verification_diagnostics", dict(self.verification_diagnostics))
        object.__setattr__(self, "hard_vetoes", _string_tuple(self.hard_vetoes))
        object.__setattr__(self, "repair_triggers", _string_tuple(self.repair_triggers))

    @property
    def passed(self) -> bool:
        return self.final_status == "passed"

    @property
    def selected_step_size(self) -> float | None:
        if self.fixed_kernel_step_size is not None:
            return self.fixed_kernel_step_size
        if self.ladder_result is None:
            return None
        value = self.ladder_result.get("selected_step_size")
        return None if value is None else float(value)

    @property
    def selected_acceptance_rate(self) -> float | None:
        return _scalar_or_none(self.verification_diagnostics.get("acceptance_rate"))

    @property
    def artifact_hash(self) -> str:
        return _stable_hash(self.payload())

    def payload(self) -> Mapping[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "handoff_source": (
                "fixed_grid_scale_probe"
                if self.fixed_kernel_step_size is not None
                else "fixed_mass_dual_averaging_ladder"
            ),
            "ladder_artifact_hash": (
                None if self.ladder_result is None else _stable_hash(self.ladder_result)
            ),
            "ladder": self.ladder_result,
            "selected_step_size": self.selected_step_size,
            "fixed_kernel_step_size": self.fixed_kernel_step_size,
            "verification_config_payload": self.verification_config_payload,
            "verification_diagnostics": self.verification_diagnostics,
            "final_status": self.final_status,
            "diagnostic_role": self.diagnostic_role,
            "hard_vetoes": self.hard_vetoes,
            "repair_triggers": self.repair_triggers,
            "passed": self.passed,
            "reports_posterior_convergence": False,
        }


@dataclass(frozen=True)
class FixedTransportHMCKernelTuningResult:
    config: FixedTransportHMCKernelTuningConfig
    transformed_adapter_signature: str
    base_adapter_signature: str
    fixed_transport_manifest_hash: str
    target_dimension: int
    identity_z_mass_artifact_payload: Mapping[str, Any]
    identity_z_mass_artifact_signature: str
    candidates: tuple[FixedTransportHMCCandidateResult, ...]
    selected_candidate_index: int | None
    final_status: str
    final_kernel_payload: Mapping[str, Any] | None
    tuning_scope_payload: Mapping[str, Any]
    route_record_payload: Mapping[str, Any]
    coordinate_payload: Mapping[str, Any]
    source_dependency_closure: Mapping[str, Any]
    candidate_selection_payload: Mapping[str, Any]
    full_chain_runner_evidence: Mapping[str, Any] | None = None
    artifact_path: str | None = None
    fixed_grid_scale_selection_payload: Mapping[str, Any] | None = None
    diagnostic_roles: Mapping[str, str] | None = None
    hard_vetoes: tuple[str, ...] = ()
    repair_triggers: tuple[str, ...] = ()
    nonclaims: tuple[str, ...] = FIXED_TRANSPORT_HMC_TUNING_NONCLAIMS

    @property
    def passed(self) -> bool:
        return self.selected_candidate_index is not None and self.final_kernel_payload is not None

    @property
    def selected_candidate(self) -> FixedTransportHMCCandidateResult | None:
        if self.selected_candidate_index is None:
            return None
        return self.candidates[int(self.selected_candidate_index)]

    @property
    def final_kernel_hash(self) -> str | None:
        return None if self.final_kernel_payload is None else _stable_hash(self.final_kernel_payload)

    @property
    def artifact_hash(self) -> str:
        return _stable_hash(self.payload())

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.fixed_transport_hmc_kernel_tuning_result.v4",
            "config": self.config.payload(),
            "tuning_scope": self.tuning_scope_payload,
            "active_route": self.route_record_payload,
            "coordinate_identity": self.coordinate_payload,
            "source_dependency_closure": self.source_dependency_closure,
            "transformed_adapter_signature": self.transformed_adapter_signature,
            "base_adapter_signature": self.base_adapter_signature,
            "fixed_transport_manifest_hash": self.fixed_transport_manifest_hash,
            "target_dimension": self.target_dimension,
            "identity_z_mass_artifact_payload": self.identity_z_mass_artifact_payload,
            "identity_z_mass_artifact_signature": self.identity_z_mass_artifact_signature,
            "candidates": tuple(candidate.payload() for candidate in self.candidates),
            "candidate_selection": self.candidate_selection_payload,
            "full_chain_runner_evidence": self.full_chain_runner_evidence,
            "selected_candidate_index": self.selected_candidate_index,
            "final_status": self.final_status,
            "final_kernel_payload": self.final_kernel_payload,
            "final_kernel_hash": self.final_kernel_hash,
            "artifact_path": self.artifact_path,
            "fixed_grid_scale_selection": self.fixed_grid_scale_selection_payload,
            "diagnostic_roles": self.diagnostic_roles or {},
            "hard_vetoes": self.hard_vetoes,
            "repair_triggers": self.repair_triggers,
            "passed": self.passed,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "reports_default_readiness": False,
            "nonclaims": self.nonclaims,
        }


@dataclass(frozen=True)
class VerifiedFixedTransportHMCHandoff:
    """Executable adapter and frozen mechanics validated against one tuning result."""

    transformed_adapter: FixedTransportValueScoreAdapter
    step_size: float
    num_leapfrog_steps: int
    handoff_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "step_size", _positive_float(self.step_size, name="step_size")
        )
        leapfrog = int(self.num_leapfrog_steps)
        if leapfrog < 2:
            raise ValueError("num_leapfrog_steps must be greater than or equal to 2")
        object.__setattr__(self, "num_leapfrog_steps", leapfrog)
        object.__setattr__(self, "handoff_payload", dict(self.handoff_payload))

    @property
    def handoff_hash(self) -> str:
        return _stable_hash(self.handoff_payload)

    def payload(self) -> Mapping[str, Any]:
        return {**self.handoff_payload, "handoff_hash": self.handoff_hash}


def build_verified_fixed_transport_hmc_handoff_from_tuning_result(
    *,
    tuning_result: FixedTransportHMCKernelTuningResult,
    base_adapter: Any,
    fixed_transport: Any,
) -> VerifiedFixedTransportHMCHandoff:
    """Rebuild and verify the exact transformed target authorized for sampling.

    The handoff binds the executable adapter to the target, transport, identity
    mass, selected mechanics, independent verification, and candidate-selection
    evidence recorded by the authoritative
    tuner. A caller cannot substitute a differently scoped adapter or reselect
    from candidate records at this boundary.
    """

    if not isinstance(tuning_result, FixedTransportHMCKernelTuningResult):
        raise TypeError("tuning_result must be FixedTransportHMCKernelTuningResult")
    if not tuning_result.passed or tuning_result.final_kernel_payload is None:
        raise ValueError("fixed-transport tuning result did not authorize a kernel")
    selected = tuning_result.selected_candidate
    if selected is None or not selected.passed:
        raise ValueError("fixed-transport tuning result lost its selected candidate")
    cfg = tuning_result.config
    target_scope = cfg.target_scope
    if not target_scope:
        raise ValueError("fixed-transport handoff requires the tuned target scope")
    adapter = build_fixed_transport_value_score_adapter(
        base_adapter=base_adapter,
        fixed_transport=fixed_transport,
        target_scope=target_scope,
        evidence_path=None,
        xla_hmc_ready=cfg.use_xla,
        full_chain_xla_diagnostic_ready=cfg.use_xla,
    )
    kernel = dict(tuning_result.final_kernel_payload)
    checks = {
        "base_adapter_signature": (
            _base_adapter_signature(base_adapter),
            tuning_result.base_adapter_signature,
        ),
        "transformed_adapter_signature": (
            adapter.adapter_signature(),
            tuning_result.transformed_adapter_signature,
        ),
        "fixed_transport_manifest_hash": (
            adapter.transport_manifest_hash,
            tuning_result.fixed_transport_manifest_hash,
        ),
        "target_dimension": (
            int(adapter.parameter_dim),
            int(tuning_result.target_dimension),
        ),
        "identity_z_mass_artifact_signature": (
            _stable_hash(tuning_result.identity_z_mass_artifact_payload),
            tuning_result.identity_z_mass_artifact_signature,
        ),
    }
    mismatches = tuple(name for name, pair in checks.items() if pair[0] != pair[1])
    if mismatches:
        raise ValueError(
            "fixed-transport handoff lineage mismatch: " + ", ".join(mismatches)
        )
    for name in (
        "base_adapter_signature",
        "transformed_adapter_signature",
        "fixed_transport_manifest_hash",
        "identity_z_mass_artifact_signature",
    ):
        if kernel.get(name) != checks[name][1]:
            raise ValueError(f"final kernel {name} mismatch")
    step = _positive_float(kernel.get("step_size"), name="final kernel step_size")
    leapfrog = int(kernel.get("num_leapfrog_steps"))
    if (
        step != selected.selected_step_size
        or leapfrog != selected.num_leapfrog_steps
    ):
        raise ValueError("final kernel mechanics do not match the selected candidate")
    acceptance = _scalar_or_none(
        kernel.get("verification_diagnostics", {}).get("acceptance_rate")
    )
    if (
        acceptance is None
        or not cfg.acceptance_band[0] <= acceptance <= cfg.acceptance_band[1]
    ):
        raise ValueError("final kernel verification acceptance is outside the pass band")
    selection = tuning_result.candidate_selection_payload
    if (
        not isinstance(selection, Mapping)
        or selection.get("final_status") != "passed"
        or selection.get("selected_candidate_index")
        != tuning_result.selected_candidate_index
    ):
        raise ValueError("fixed-transport tuning lacks a matching passed selection")
    if kernel.get("tuning_scope") != tuning_result.tuning_scope_payload:
        raise ValueError("final kernel tuning scope mismatch")
    if kernel.get("candidate_selection_hash") != _stable_hash(selection):
        raise ValueError("final kernel candidate selection hash mismatch")
    if kernel.get("source_dependency_closure_hash") != _stable_hash(
        tuning_result.source_dependency_closure
    ):
        raise ValueError("final kernel source dependency closure mismatch")
    if (
        kernel.get("full_chain_runner_evidence")
        != tuning_result.full_chain_runner_evidence
    ):
        raise ValueError("final kernel full-chain runner evidence mismatch")
    if tuning_result.source_dependency_closure != _tuning_source_dependency_closure():
        raise ValueError("tuning source dependency closure no longer matches runtime")
    if selection.get("transformed_adapter_signature") != adapter.adapter_signature():
        raise ValueError("candidate selection adapter signature mismatch")
    if selection.get("target_scope") != target_scope:
        raise ValueError("candidate selection target scope mismatch")
    selected_rows = tuple(
        row
        for row in selection.get("candidate_rows", ())
        if isinstance(row, Mapping)
        and row.get("candidate_index") == tuning_result.selected_candidate_index
    )
    if len(selected_rows) != 1:
        raise ValueError("candidate selection lacks one selected evidence row")
    selected_row = selected_rows[0]
    if (
        selected_row.get("eligible") is not True
        or selected_row.get("candidate_artifact_hash") != selected.artifact_hash
    ):
        raise ValueError("selected candidate evidence binding mismatch")
    if cfg.selection_policy == "replicated_min_bulk_ess_per_gradient":
        evidence = selected_row.get("selection_evidence")
        if not isinstance(evidence, Mapping) or evidence.get("final_status") != "passed":
            raise ValueError("selected candidate lacks passed efficiency evidence")
        heldout = selection.get("heldout_verification")
        if (
            not isinstance(heldout, Mapping)
            or heldout.get("final_status") != "passed"
            or selection.get("post_selection_candidate_only_verification_used")
            is not True
            or selection.get("candidate_verification_serves_as_final") is not False
        ):
            raise ValueError("selected candidate lacks passed held-out verification")
        if kernel.get("verification_diagnostics") != heldout.get("diagnostics"):
            raise ValueError("final kernel held-out diagnostics mismatch")
        if kernel.get("heldout_verification_hash") != _stable_hash(heldout):
            raise ValueError("final kernel held-out verification hash mismatch")
        heldout_config = heldout.get("config_payload")
        if (
            not isinstance(heldout_config, Mapping)
            or float(heldout_config.get("step_size", math.nan)) != step
            or int(heldout_config.get("num_leapfrog_steps", -1)) != leapfrog
            or heldout_config.get("adaptation_policy")
            != "fixed_kernel_no_adaptation"
        ):
            raise ValueError("held-out verification mechanics mismatch")
        if cfg.require_modern_rank_normalized_verification:
            modern = heldout.get("diagnostics", {}).get(
                "modern_rank_normalized_verification"
            )
            if not isinstance(modern, Mapping) or modern.get("passed") is not True:
                raise ValueError("held-out modern rank-normalized verification failed")
    seed_ledger = selection.get("seed_ledger")
    if not isinstance(seed_ledger, Mapping) or seed_ledger.get("all_seeds_unique") is not True:
        raise ValueError("candidate selection lacks a disjoint seed ledger")
    seed_rows = seed_ledger.get("rows", ())
    seeds = tuple(
        tuple(int(value) for value in row.get("seed", ()))
        for row in seed_rows
        if isinstance(row, Mapping)
    )
    if (
        len(seeds) != int(seed_ledger.get("seed_count", -1))
        or any(len(seed) != 2 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("candidate selection seed ledger is inconsistent")
    payload = {
        "schema": "bayesfilter.verified_fixed_transport_hmc_handoff.v1",
        "runtime": "bayesfilter.inference.fixed_transport_hmc_tuning_tf",
        "tuning_result_hash": tuning_result.artifact_hash,
        "final_kernel_hash": tuning_result.final_kernel_hash,
        "selected_candidate_index": tuning_result.selected_candidate_index,
        "selection_policy": cfg.selection_policy,
        "tuning_scope": tuning_result.tuning_scope_payload,
        "candidate_selection_hash": _stable_hash(selection),
        "heldout_verification_hash": kernel.get("heldout_verification_hash"),
        "step_size": step,
        "num_leapfrog_steps": leapfrog,
        "mass_policy": "fixed_identity_z",
        "base_adapter_signature": tuning_result.base_adapter_signature,
        "transformed_adapter_signature": tuning_result.transformed_adapter_signature,
        "fixed_transport_manifest_hash": tuning_result.fixed_transport_manifest_hash,
        "identity_z_mass_artifact_signature": (
            tuning_result.identity_z_mass_artifact_signature
        ),
        "target_scope": target_scope,
        "target_dimension": tuning_result.target_dimension,
        "use_xla": cfg.use_xla,
        "tuning_draws_discarded": True,
        "nonclaims": FIXED_TRANSPORT_HMC_TUNING_NONCLAIMS,
    }
    return VerifiedFixedTransportHMCHandoff(
        transformed_adapter=adapter,
        step_size=step,
        num_leapfrog_steps=leapfrog,
        handoff_payload=payload,
    )


def tune_fixed_transport_hmc_kernel(
    *,
    base_adapter: Any,
    fixed_transport: Any,
    initial_position: Any,
    config: FixedTransportHMCKernelTuningConfig | None = None,
    output_dir: str | Path | None = None,
    run_full_chain: RunFullChainFn = _run_full_chain_tfp_hmc,
    passthrough_exceptions: tuple[type[Exception], ...] = (),
) -> FixedTransportHMCKernelTuningResult:
    """Tune fixed-length TFP HMC and verify a frozen identity-`z` kernel.

    Campaign-owned resource exceptions may be passed through so a caller can
    close out as under-budgeted instead of recording a synthetic numerical
    tuning veto. All other runtime exceptions retain the fail-closed artifact
    behavior.
    """

    route_record = require_active_hmc_tuning_route(
        "tune_fixed_transport_hmc_kernel"
    )
    cfg = config or FixedTransportHMCKernelTuningConfig(initial_step_size=0.1)
    if not isinstance(cfg, FixedTransportHMCKernelTuningConfig):
        raise TypeError("config must be FixedTransportHMCKernelTuningConfig")
    if (
        run_full_chain is _run_full_chain_tfp_hmc
        and cfg.chain_execution_mode == "tf_function"
    ):
        run_full_chain = FixedTransportReusableRunnerPool()
    passthrough = tuple(passthrough_exceptions)
    if any(
        not isinstance(exception_type, type)
        or not issubclass(exception_type, Exception)
        for exception_type in passthrough
    ):
        raise TypeError("passthrough_exceptions must contain Exception subclasses")
    capability = value_score_capability(base_adapter)
    if capability.value_score_authority in _FORBIDDEN_BASE_AUTHORITIES:
        raise ValueError("fixed-transport HMC tuning forbids gradient_tape_fallback")
    target_scope = cfg.target_scope or (
        None
        if capability.target_scope is None
        else f"{capability.target_scope}:fixed_transport"
    )
    if not target_scope:
        raise ValueError("target_scope is required when the base adapter has none")
    cfg = replace(cfg, target_scope=target_scope)
    adapter = build_fixed_transport_value_score_adapter(
        base_adapter=base_adapter,
        fixed_transport=fixed_transport,
        target_scope=target_scope,
        evidence_path=None,
        xla_hmc_ready=cfg.use_xla,
        full_chain_xla_diagnostic_ready=cfg.use_xla,
    )
    z0 = _validate_initial_position(initial_position, adapter.parameter_dim)
    transformed_signature = adapter.adapter_signature()
    base_signature = _base_adapter_signature(base_adapter)
    coordinate_payload = {
        "schema": "bayesfilter.fixed_transport_hmc_coordinate_identity.v1",
        "coordinate_system": "fixed_transport_latent_z",
        "mass_policy": "fixed_identity_z",
        "fixed_transport_manifest_hash": adapter.transport_manifest_hash,
        "parameter_dimension": adapter.parameter_dim,
    }
    tuning_scope = HMCTuningScope(
        target_scope=target_scope,
        adapter_signature=transformed_signature,
        coordinate_signature=_stable_hash(coordinate_payload),
        transport_signature=adapter.transport_manifest_hash,
        parameter_dimension=adapter.parameter_dim,
        backend="tensorflow_probability",
        dtype="float64",
        xla_enabled=cfg.use_xla,
        chain_execution_mode=cfg.chain_execution_mode,
    )
    source_closure = _tuning_source_dependency_closure()
    mass_payload = _identity_mass_payload(z0, transformed_signature)
    mass_signature = _stable_hash(mass_payload)
    candidates, scale_payload = _candidate_attempts(
        cfg,
        adapter=adapter,
        z0=z0,
        run_full_chain=run_full_chain,
        passthrough_exceptions=passthrough,
    )
    selection = _select_candidates(
        candidates,
        cfg,
        adapter=adapter,
        z0=z0,
        run_full_chain=run_full_chain,
        passthrough_exceptions=passthrough,
    )
    evidence_fn = getattr(run_full_chain, "evidence", None)
    runner_evidence = evidence_fn() if callable(evidence_fn) else None
    if (
        isinstance(runner_evidence, Mapping)
        and runner_evidence.get("all_runners_traced_exactly_once") is not True
    ):
        raise RuntimeError(
            "fixed-transport reusable runner did not trace every static graph "
            "exactly once: "
            + json.dumps(_json_ready(runner_evidence), sort_keys=True)
        )
    selected_raw = selection.get("selected_candidate_index")
    selected = None if selected_raw is None else int(selected_raw)
    selected_candidate = None if selected is None else candidates[selected]
    final_kernel = None
    if selected_candidate is not None:
        heldout = selection.get("heldout_verification")
        if cfg.selection_policy == "replicated_min_bulk_ess_per_gradient":
            if not isinstance(heldout, Mapping) or heldout.get("final_status") != "passed":
                raise RuntimeError("passed efficiency selection lacks held-out verification")
            verification_diagnostics = heldout["diagnostics"]
        else:
            verification_diagnostics = selected_candidate.verification_diagnostics
        selected_selection_row = next(
            row
            for row in selection["candidate_rows"]
            if row["candidate_index"] == selected
        )
        selection_evidence = selected_selection_row.get("selection_evidence")
        final_kernel = {
            "schema": "bayesfilter.fixed_transport_hmc_kernel.v4",
            "runtime": "bayesfilter.inference.tune_fixed_transport_hmc_kernel",
            "runtime_numerical_backend": "tensorflow_tfp_only",
            "step_size": selected_candidate.selected_step_size,
            "num_leapfrog_steps": selected_candidate.num_leapfrog_steps,
            "mass_policy": "fixed_identity_z",
            "identity_z_mass_artifact_payload": mass_payload,
            "identity_z_mass_artifact_signature": mass_signature,
            "transformed_adapter_signature": transformed_signature,
            "base_adapter_signature": base_signature,
            "fixed_transport_manifest_hash": adapter.transport_manifest_hash,
            "tuning_scope": tuning_scope.payload(),
            "active_route": route_record.payload(),
            "coordinate_identity": coordinate_payload,
            "source_dependency_closure_hash": _stable_hash(source_closure),
            "full_chain_runner_evidence": runner_evidence,
            "candidate_selection_hash": _stable_hash(selection),
            "heldout_verification_hash": (
                None if not isinstance(heldout, Mapping) else _stable_hash(heldout)
            ),
            "rank2_chain_batched_target_required": True,
            "shared_scalar_step_across_chain_bank": True,
            "use_xla": cfg.use_xla,
            "proposal_dynamics_identity": cfg.proposal_dynamics_identity,
            "windowed_mass_adaptation_used": False,
            "mass_adaptation_used": False,
            "transport_training_or_adaptation_used": False,
            "selection_policy": cfg.selection_policy,
            "candidate_selection": selection,
            "candidate_selection_diagnostics": (
                None
                if not isinstance(selection_evidence, Mapping)
                else selection_evidence.get("diagnostics")
            ),
            "verification_diagnostics": verification_diagnostics,
            "fresh_candidate_verification_used": (
                cfg.selection_policy == "acceptance_target_distance"
            ),
            "post_selection_candidate_only_verification_used": (
                cfg.selection_policy == "replicated_min_bulk_ess_per_gradient"
            ),
            "nonclaims": FIXED_TRANSPORT_HMC_TUNING_NONCLAIMS,
        }
    final_status = str(selection.get("final_status", "no_viable_candidate"))
    selection_hard_vetoes = tuple(selection.get("hard_vetoes", ()))
    selection_repair_triggers = tuple(selection.get("repair_triggers", ()))
    result = FixedTransportHMCKernelTuningResult(
        config=cfg,
        transformed_adapter_signature=transformed_signature,
        base_adapter_signature=base_signature,
        fixed_transport_manifest_hash=adapter.transport_manifest_hash,
        target_dimension=adapter.parameter_dim,
        identity_z_mass_artifact_payload=mass_payload,
        identity_z_mass_artifact_signature=mass_signature,
        candidates=tuple(candidates),
        selected_candidate_index=selected,
        final_status=final_status,
        final_kernel_payload=final_kernel,
        tuning_scope_payload=tuning_scope.payload(),
        route_record_payload=route_record.payload(),
        coordinate_payload=coordinate_payload,
        source_dependency_closure=source_closure,
        candidate_selection_payload=selection,
        full_chain_runner_evidence=runner_evidence,
        fixed_grid_scale_selection_payload=scale_payload,
        diagnostic_roles=_diagnostic_roles(),
        hard_vetoes=tuple(
            dict.fromkeys(
                [
                    veto
                    for candidate in candidates
                    for veto in candidate.hard_vetoes
                ]
                + list(selection_hard_vetoes)
            )
        ),
        repair_triggers=tuple(
            dict.fromkeys(
                [
                    trigger
                    for candidate in candidates
                    for trigger in candidate.repair_triggers
                ]
                + list(selection_repair_triggers)
            )
        ),
    )
    if output_dir is None:
        return result
    path = Path(output_dir) / cfg.output_filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"tuning artifact already exists: {path}")
    result = FixedTransportHMCKernelTuningResult(
        **{**result.__dict__, "artifact_path": str(path)}
    )
    path.write_text(
        json.dumps(_json_ready(result.payload()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _candidate_attempts(
    config: FixedTransportHMCKernelTuningConfig,
    *,
    adapter: FixedTransportValueScoreAdapter,
    z0: tf.Tensor,
    run_full_chain: RunFullChainFn,
    passthrough_exceptions: tuple[type[Exception], ...],
) -> tuple[list[FixedTransportHMCCandidateResult], Mapping[str, Any] | None]:
    if config.fixed_grid_base_step_size_candidates:
        return _fixed_grid_attempts(
            config,
            adapter=adapter,
            z0=z0,
            run_full_chain=run_full_chain,
            passthrough_exceptions=passthrough_exceptions,
        )
    return [
        _dual_averaging_candidate(
            config,
            adapter=adapter,
            z0=z0,
            candidate_index=index,
            leapfrog=leapfrog,
            run_full_chain=run_full_chain,
            passthrough_exceptions=passthrough_exceptions,
        )
        for index, leapfrog in enumerate(config.leapfrog_grid)
    ], None


def _dual_averaging_candidate(
    config: FixedTransportHMCKernelTuningConfig,
    *,
    adapter: FixedTransportValueScoreAdapter,
    z0: tf.Tensor,
    candidate_index: int,
    leapfrog: int,
    run_full_chain: RunFullChainFn,
    passthrough_exceptions: tuple[type[Exception], ...],
) -> FixedTransportHMCCandidateResult:
    state = _initial_state(config, adapter.parameter_dim, z0=z0)
    step = config.initial_step_size
    rounds = []
    selected_step = None
    repair_triggers: list[str] = []
    hard_vetoes: list[str] = []
    screen_failure_vetoes: list[str] = []
    for round_index, budget in enumerate(config.budget_schedule):
        round_initial_step = step
        tune_config = _chain_config(
            config,
            num_results=config.tune_num_results,
            burnin=budget,
            step=step,
            leapfrog=leapfrog,
            seed=_derived_seed(
                config.tune_seed_base,
                1,
                candidate_index,
                round_index,
            ),
            adaptive_steps=budget,
        )
        try:
            tune_result = run_full_chain(adapter, state, tune_config)
            tune_diagnostics = _verification_diagnostics(
                tune_result,
                adapter=adapter,
                config=config,
                initial_state=state,
                require_modern=False,
            )
            tuned_step = _scalar_or_none(tune_result.diagnostics.get("final_step_size"))
        except Exception as exc:  # noqa: BLE001 - produce a fail-closed artifact.
            if isinstance(exc, passthrough_exceptions):
                raise
            tune_diagnostics = _error_diagnostics(exc)
            tuned_step = None
        tune_vetoes = _basic_hard_vetoes(tune_diagnostics, prefix="tune")
        if tuned_step is None or not math.isfinite(tuned_step) or tuned_step <= 0.0:
            tune_vetoes.append("tune_step_missing_or_nonfinite")
        screen_diagnostics: Mapping[str, Any] = {}
        screen_vetoes: tuple[str, ...] = ()
        round_repairs: tuple[str, ...] = ()
        if not tune_vetoes:
            screen = _run_verification(
                config,
                adapter=adapter,
                z0=z0,
                step=float(tuned_step),
                leapfrog=leapfrog,
                candidate_index=candidate_index * 100 + round_index,
                run_full_chain=run_full_chain,
                probe_only=True,
                passthrough_exceptions=passthrough_exceptions,
            )
            screen_diagnostics = screen["diagnostics"]
            screen_vetoes = screen["hard_vetoes"]
            screen_failure_vetoes.extend(screen_vetoes)
            acceptance = _scalar_or_none(screen_diagnostics.get("acceptance_rate"))
            if not screen_vetoes:
                selected_step = float(tuned_step)
            elif acceptance is not None and math.isfinite(acceptance):
                if acceptance < config.acceptance_band[0]:
                    step = float(tuned_step) / config.step_repair_factor
                    round_repairs = (
                        "screen_acceptance_below_repair_band"
                        if acceptance < config.repair_band[0]
                        else "screen_acceptance_below_pass_band"
                    ,)
                elif acceptance > config.acceptance_band[1]:
                    step = float(tuned_step) * config.step_repair_factor
                    round_repairs = (
                        "screen_acceptance_above_repair_band"
                        if acceptance > config.repair_band[1]
                        else "screen_acceptance_above_pass_band"
                    ,)
        rounds.append(
            {
                "round_index": round_index,
                "budget": budget,
                "initial_step_size": round_initial_step,
                "tuned_step_size": tuned_step,
                "tune_config": tune_config.signature_payload(),
                "tune_diagnostics": tune_diagnostics,
                "screen_config": (
                    None if not screen_diagnostics else screen["config_payload"]
                ),
                "screen_diagnostics": screen_diagnostics,
                "hard_vetoes": tuple(tune_vetoes) + tuple(screen_vetoes),
                "repair_triggers": round_repairs,
            }
        )
        repair_triggers.extend(round_repairs)
        if tune_vetoes:
            hard_vetoes.extend(tune_vetoes)
            break
        if selected_step is not None:
            break
    ladder = {
        "schema": "bayesfilter.fixed_transport_hmc_tf_ladder.v1",
        "rounds": tuple(rounds),
        "selected_step_size": selected_step,
        "passed": selected_step is not None,
        "shared_scalar_step_across_chain_bank": True,
        "runtime_numerical_backend": "tensorflow_tfp_only",
    }
    if selected_step is None:
        hard_vetoes.extend(screen_failure_vetoes)
        return FixedTransportHMCCandidateResult(
            candidate_index,
            leapfrog,
            ladder,
            None,
            {},
            "ladder_no_viable_step",
            "ladder_nonpass",
            hard_vetoes=tuple(hard_vetoes),
            repair_triggers=tuple(repair_triggers),
        )
    if config.selection_policy == "replicated_min_bulk_ess_per_gradient":
        final_round = rounds[-1]
        return FixedTransportHMCCandidateResult(
            candidate_index,
            leapfrog,
            ladder,
            final_round["screen_config"],
            final_round["screen_diagnostics"],
            "passed",
            "fixed_transport_step_ladder_nomination",
            hard_vetoes=(),
            repair_triggers=tuple(repair_triggers),
        )
    verification = _run_verification(
        config,
        adapter=adapter,
        z0=z0,
        step=selected_step,
        leapfrog=leapfrog,
        candidate_index=candidate_index,
        run_full_chain=run_full_chain,
        probe_only=False,
        passthrough_exceptions=passthrough_exceptions,
    )
    return FixedTransportHMCCandidateResult(
        candidate_index,
        leapfrog,
        ladder,
        verification["config_payload"],
        verification["diagnostics"],
        verification["final_status"],
        verification["diagnostic_role"],
        hard_vetoes=verification["hard_vetoes"],
        repair_triggers=tuple(repair_triggers) + verification["repair_triggers"],
    )


def _run_replicated_efficiency_selection(
    config: FixedTransportHMCKernelTuningConfig,
    *,
    adapter: FixedTransportValueScoreAdapter,
    z0: tf.Tensor,
    step: float,
    leapfrog: int,
    candidate_index: int,
    run_full_chain: RunFullChainFn,
    passthrough_exceptions: tuple[type[Exception], ...],
) -> Mapping[str, Any]:
    """Run fixed-kernel nomination screens and aggregate a conservative score.

    The minimum replication-level bulk-ESS-per-declared-gradient score prevents
    one unusually favorable short chain from winning the trajectory grid. These
    screens remain discarded heuristics. Every input candidate has already
    passed an independent in-band fixed-kernel ladder screen, but has not yet
    consumed the post-selection held-out verification.
    """

    rows = []
    hard_vetoes: list[str] = []
    repair_triggers: list[str] = []
    gradient_count = (
        config.chain_count
        * (config.selection_num_burnin_steps + config.selection_num_results)
        * int(leapfrog)
    )
    for replication in range(config.selection_replications):
        check = _run_verification(
            config,
            adapter=adapter,
            z0=z0,
            step=step,
            leapfrog=leapfrog,
            candidate_index=candidate_index,
            run_full_chain=run_full_chain,
            probe_only=False,
            passthrough_exceptions=passthrough_exceptions,
            selection_replication=replication,
        )
        diagnostics = dict(check["diagnostics"])
        efficiency = diagnostics.get("selection_efficiency_diagnostics")
        replication_vetoes = list(check["hard_vetoes"])
        min_bulk_ess = None
        maximum_rhat = None
        score = None
        if isinstance(efficiency, Mapping):
            min_bulk_ess = _scalar_or_none(efficiency.get("min_bulk_ess"))
            maximum_rhat = _scalar_or_none(efficiency.get("max_rhat"))
        if (
            min_bulk_ess is None
            or maximum_rhat is None
            or not math.isfinite(min_bulk_ess)
            or not math.isfinite(maximum_rhat)
            or min_bulk_ess <= 0.0
        ):
            replication_vetoes.append("selection_efficiency_missing_or_nonfinite")
        else:
            score = min_bulk_ess / float(gradient_count)
        mean_esjd = _scalar_or_none(
            diagnostics.get("mean_squared_jump_distance_hmc_coordinates")
        )
        esjd_per_gradient = (
            None
            if mean_esjd is None or not math.isfinite(mean_esjd)
            else mean_esjd / float(leapfrog)
        )
        prefixed_vetoes = tuple(
            f"selection_replication_{replication}_{reason}"
            for reason in dict.fromkeys(replication_vetoes)
        )
        prefixed_repairs = tuple(
            f"selection_replication_{replication}_{reason}"
            for reason in check["repair_triggers"]
        )
        hard_vetoes.extend(prefixed_vetoes)
        repair_triggers.extend(prefixed_repairs)
        rows.append(
            {
                "replication": replication,
                "config_payload": check["config_payload"],
                "diagnostics": diagnostics,
                "acceptance_rate": _scalar_or_none(
                    diagnostics.get("acceptance_rate")
                ),
                "minimum_bulk_ess": min_bulk_ess,
                "maximum_modern_rhat": maximum_rhat,
                "declared_target_gradient_count": gradient_count,
                "minimum_bulk_ess_per_declared_target_gradient": score,
                "mean_esjd_per_retained_transition_gradient": esjd_per_gradient,
                "hard_vetoes": prefixed_vetoes,
                "repair_triggers": prefixed_repairs,
                "passed": not prefixed_vetoes,
            }
        )
    scores = tuple(
        float(row["minimum_bulk_ess_per_declared_target_gradient"])
        for row in rows
        if row["minimum_bulk_ess_per_declared_target_gradient"] is not None
    )
    rhats = tuple(
        float(row["maximum_modern_rhat"])
        for row in rows
        if row["maximum_modern_rhat"] is not None
    )
    acceptances = tuple(
        float(row["acceptance_rate"])
        for row in rows
        if row["acceptance_rate"] is not None
    )
    passed = not hard_vetoes and len(scores) == config.selection_replications
    diagnostics = {
        "schema": "bayesfilter.fixed_transport_hmc_replicated_selection.v1",
        "selection_policy": config.selection_policy,
        "selection_replication_count": config.selection_replications,
        "replications": tuple(rows),
        "acceptance_rate": (
            None
            if len(acceptances) != config.selection_replications
            else sum(acceptances) / float(len(acceptances))
        ),
        "acceptance_rate_by_replication": acceptances,
        "minimum_bulk_ess_per_declared_target_gradient": (
            min(scores) if len(scores) == config.selection_replications else None
        ),
        "maximum_modern_rhat_across_replications": (
            max(rhats) if len(rhats) == config.selection_replications else None
        ),
        "declared_target_gradient_count_per_replication": gradient_count,
        "gradient_count_semantics": (
            "chain_count * (discarded_burnin + discarded_results) * L"
        ),
        "esjd_rate_semantics": (
            "mean retained-transition squared jump distance divided by L; "
            "burn-in jumps are unavailable"
        ),
        "all_selection_draws_discarded": True,
        "passed": passed,
        "reports_posterior_convergence": False,
        "reports_sampler_superiority": False,
    }
    return {
        "config_payload": {
            "schema": "bayesfilter.fixed_transport_hmc_selection_configs.v1",
            "replications": tuple(row["config_payload"] for row in rows),
        },
        "diagnostics": diagnostics,
        "final_status": "passed" if passed else "hard_veto",
        "diagnostic_role": (
            "replicated_candidate_selection_screen" if passed else "hard_veto"
        ),
        "hard_vetoes": tuple(dict.fromkeys(hard_vetoes)),
        "repair_triggers": tuple(dict.fromkeys(repair_triggers)),
    }


def _fixed_grid_attempts(
    config: FixedTransportHMCKernelTuningConfig,
    *,
    adapter: FixedTransportValueScoreAdapter,
    z0: tf.Tensor,
    run_full_chain: RunFullChainFn,
    passthrough_exceptions: tuple[type[Exception], ...],
) -> tuple[list[FixedTransportHMCCandidateResult], Mapping[str, Any]]:
    leapfrog = config.fixed_grid_num_leapfrog_steps or max(config.leapfrog_grid)
    base = max(config.fixed_grid_base_step_size_candidates)
    attempts = []
    selected_step = None
    for index, scale in enumerate(
        config.fixed_grid_scale_candidates[: config.fixed_grid_max_attempts]
    ):
        step = base * scale
        probe = _run_verification(
            config,
            adapter=adapter,
            z0=z0,
            step=step,
            leapfrog=leapfrog,
            candidate_index=10_000 + index,
            run_full_chain=run_full_chain,
            probe_only=True,
            passthrough_exceptions=passthrough_exceptions,
        )
        acceptance = _scalar_or_none(probe["diagnostics"].get("acceptance_rate"))
        acceptance_class = _acceptance_class(acceptance, config)
        attempts.append(
            {
                "attempt_index": index,
                "scale": scale,
                "initial_step_size": step,
                "pilot_num_leapfrog_steps": leapfrog,
                "pilot_acceptance_rate": acceptance,
                "acceptance_class": acceptance_class,
                "probe_final_status": probe["final_status"],
                "probe_diagnostic_role": probe["diagnostic_role"],
                "probe_config_payload": probe["config_payload"],
                "probe_hard_vetoes": probe["hard_vetoes"],
                "probe_repair_triggers": probe["repair_triggers"],
                "probe_diagnostics": probe["diagnostics"],
            }
        )
        if acceptance_class == "in_band" and not probe["hard_vetoes"]:
            selected_step = step
            break
    scale_payload = {
        "artifact_type": "bayesfilter_fixed_transport_hmc_grid_scale_repair",
        "schema_version": 2,
        "attempts": tuple(attempts),
        "selected_scale": None if selected_step is None else attempts[-1]["scale"],
        "status": "accepted_in_band" if selected_step is not None else "repair_attempts_exhausted",
        "nonclaims": FIXED_TRANSPORT_HMC_TUNING_NONCLAIMS,
    }
    if selected_step is None:
        return [], scale_payload
    if config.selection_policy == "replicated_min_bulk_ess_per_gradient":
        final_probe = attempts[-1]
        candidate = FixedTransportHMCCandidateResult(
            0,
            leapfrog,
            None,
            final_probe["probe_config_payload"],
            final_probe["probe_diagnostics"],
            "passed",
            "fixed_transport_step_ladder_nomination",
            fixed_kernel_step_size=selected_step,
            hard_vetoes=(),
            repair_triggers=tuple(final_probe["probe_repair_triggers"]),
        )
        return [candidate], scale_payload
    verification = _run_verification(
        config,
        adapter=adapter,
        z0=z0,
        step=selected_step,
        leapfrog=leapfrog,
        candidate_index=0,
        run_full_chain=run_full_chain,
        probe_only=False,
        passthrough_exceptions=passthrough_exceptions,
    )
    candidate = FixedTransportHMCCandidateResult(
        0,
        leapfrog,
        None,
        verification["config_payload"],
        verification["diagnostics"],
        verification["final_status"],
        verification["diagnostic_role"],
        fixed_kernel_step_size=selected_step,
        hard_vetoes=verification["hard_vetoes"],
        repair_triggers=verification["repair_triggers"],
    )
    return [candidate], scale_payload


def _run_verification(
    config: FixedTransportHMCKernelTuningConfig,
    *,
    adapter: FixedTransportValueScoreAdapter,
    z0: tf.Tensor,
    step: float,
    leapfrog: int,
    candidate_index: int,
    run_full_chain: RunFullChainFn,
    probe_only: bool,
    passthrough_exceptions: tuple[type[Exception], ...],
    selection_replication: int | None = None,
    post_selection_heldout: bool = False,
) -> Mapping[str, Any]:
    if sum(
        (bool(probe_only), selection_replication is not None, bool(post_selection_heldout))
    ) > 1:
        raise ValueError("verification roles must be mutually exclusive")
    require_complete_transition_telemetry = False
    final_verification = not probe_only and selection_replication is None
    require_modern = bool(
        config.require_modern_rank_normalized_verification and final_verification
    )
    report_modern = bool(
        config.report_modern_rank_normalized_verification and final_verification
    )
    if post_selection_heldout:
        num_results = config.verification_num_results
        burnin = config.verification_num_burnin_steps
        seed = _derived_seed(config.verification_seed_base, 4, candidate_index)
        diagnostic_context = "fixed_transport_post_selection_heldout_verification"
        require_efficiency = False
        require_complete_transition_telemetry = True
    elif selection_replication is not None:
        num_results = config.selection_num_results
        burnin = config.selection_num_burnin_steps
        seed = _derived_seed(
            config.selection_seed_base,
            3,
            candidate_index,
            int(selection_replication),
        )
        diagnostic_context = "fixed_transport_candidate_selection_replication"
        require_efficiency = True
        require_complete_transition_telemetry = True
    elif probe_only:
        num_results = config.screen_num_results
        burnin = config.screen_num_burnin_steps
        seed = _derived_seed(config.screen_seed_base, 2, candidate_index)
        diagnostic_context = "fixed_transport_step_ladder_screen"
        require_efficiency = False
    else:
        num_results = config.verification_num_results
        burnin = config.verification_num_burnin_steps
        seed = _derived_seed(config.verification_seed_base, 4, candidate_index)
        diagnostic_context = "fixed_transport_candidate_verification"
        require_efficiency = False
    chain_config = _chain_config(
        config,
        num_results=num_results,
        burnin=burnin,
        step=step,
        leapfrog=leapfrog,
        seed=seed,
        adaptive_steps=0,
    )
    initial_state = _initial_state(config, adapter.parameter_dim, z0=z0)
    error = None
    try:
        result = run_full_chain(adapter, initial_state, chain_config)
        diagnostics = _verification_diagnostics(
            result,
            adapter=adapter,
            config=config,
            initial_state=initial_state,
            require_modern=require_modern,
            report_modern=report_modern,
            require_efficiency=require_efficiency,
        )
    except Exception as exc:  # noqa: BLE001 - verification fails closed.
        if isinstance(exc, passthrough_exceptions):
            raise
        error = exc
        diagnostics = _error_diagnostics(exc)
    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "diagnostic_context": (
                diagnostic_context
            ),
            "probe_only": probe_only,
            "selection_replication": selection_replication,
            "initial_state_shape": tuple(int(value) for value in initial_state.shape),
            "initial_state_all_zero": bool(tf.reduce_all(initial_state == 0.0).numpy()),
            "initial_state_bank": _json_ready(initial_state),
            "rank2_chain_batched_initial_state": True,
            "modern_rank_normalized_verification_role": (
                "required_hard_gate"
                if require_modern
                else (
                    "diagnostic_only_not_handoff_gate"
                    if report_modern
                    else "not_requested"
                )
            ),
            "nonclaims": FIXED_TRANSPORT_HMC_TUNING_NONCLAIMS,
        }
    )
    status, role, vetoes, repairs = _classify_verification(
        config,
        diagnostics=diagnostics,
        run_error=error,
        require_modern=require_modern,
        require_efficiency=require_efficiency,
        require_complete_transition_telemetry=require_complete_transition_telemetry,
        acceptance_band=(
            config.selection_acceptance_band
            if selection_replication is not None
            else config.acceptance_band
        ),
        diagnostic_role=(
            "post_selection_heldout_verification"
            if post_selection_heldout
            else (
                "replicated_candidate_selection_screen"
                if selection_replication is not None
                else (
                "fixed_transport_step_ladder_screen"
                if probe_only
                else "fresh_fixed_kernel_candidate_verification"
                )
            )
        ),
    )
    return {
        "config_payload": chain_config.signature_payload(),
        "diagnostics": diagnostics,
        "final_status": status,
        "diagnostic_role": role,
        "hard_vetoes": vetoes,
        "repair_triggers": repairs,
    }


def _verification_diagnostics(
    result: Any,
    *,
    adapter: FixedTransportValueScoreAdapter,
    config: FixedTransportHMCKernelTuningConfig,
    initial_state: tf.Tensor,
    require_modern: bool,
    report_modern: bool = False,
    require_efficiency: bool = False,
) -> Mapping[str, Any]:
    payload = _tensor_diagnostics(result.samples, result.trace)
    run_diagnostics = dict(result.diagnostics)
    divergence_status = run_diagnostics.get(
        "divergence_status", run_diagnostics.get("native_divergence_status")
    )
    if divergence_status is not None:
        payload["divergence_status"] = str(divergence_status)
        payload["divergence_count"] = _int_or_none(
            run_diagnostics.get("divergence_count")
        )
        payload["native_divergence_interpretation"] = (
            "available native boolean/count"
            if str(divergence_status) == "available"
            else "unavailable is not zero divergences"
        )
    if "target_status_telemetry" in run_diagnostics:
        payload["target_status_telemetry"] = _json_ready(
            run_diagnostics["target_status_telemetry"]
        )
    maximum = _scalar_or_none(payload.get("max_abs_log_accept_energy_proxy"))
    payload["log_accept_energy_proxy_alert"] = bool(
        maximum is not None
        and math.isfinite(maximum)
        and maximum > config.maximum_absolute_energy_error
    )
    payload["runtime_metadata"] = _json_ready(result.metadata)
    samples = tf.cast(tf.convert_to_tensor(result.samples), tf.float64)
    sample_shape = tuple(int(value) for value in tf.shape(samples).numpy().tolist())
    payload["sample_shape"] = sample_shape
    state = tf.cast(tf.convert_to_tensor(initial_state), tf.float64)
    state_shape = tuple(int(value) for value in tf.shape(state).numpy().tolist())
    if len(sample_shape) == 3 and sample_shape[1:] == state_shape:
        trajectory = tf.concat((state[tf.newaxis, :, :], samples), axis=0)
        increments = trajectory[1:] - trajectory[:-1]
        maximum_displacement = tf.reduce_max(
            tf.abs(samples - state[tf.newaxis, :, :]), axis=(0, 2)
        )
        moved = maximum_displacement > 0.0
        payload["maximum_displacement_by_chain"] = tuple(
            float(value) for value in maximum_displacement.numpy().tolist()
        )
        payload["chain_movement_by_chain"] = tuple(
            bool(value) for value in moved.numpy().tolist()
        )
        payload["all_chains_moved"] = bool(tf.reduce_all(moved).numpy())
        payload["mean_squared_jump_distance_hmc_coordinates"] = float(
            tf.reduce_mean(tf.reduce_sum(tf.square(increments), axis=-1)).numpy()
        )
    else:
        payload["maximum_displacement_by_chain"] = None
        payload["chain_movement_by_chain"] = None
        payload["all_chains_moved"] = False
        payload["mean_squared_jump_distance_hmc_coordinates"] = None
    payload["modern_rank_normalized_verification"] = None
    payload["modern_verification_coordinate_system"] = None
    payload["selection_efficiency_diagnostics"] = None
    if require_modern or report_modern or require_efficiency:
        raw = (
            samples
            if config.verification_coordinate_system == "hmc_coordinates"
            else _map_samples(adapter, samples)
        )
    if require_modern:
        modern = rank_normalized_split_rhat_summary(
            raw, rhat_max=config.verification_rhat_max
        )
        payload["modern_rank_normalized_verification"] = modern
        payload["modern_verification_coordinate_system"] = config.verification_coordinate_system
    elif report_modern:
        try:
            modern = rank_normalized_split_rhat_summary(
                raw, rhat_max=config.verification_rhat_max
            )
        except Exception as error:  # Diagnostic failure cannot veto mechanics.
            modern = {
                "schema": "bayesfilter.rank_normalized_split_rhat_diagnostic_error.v1",
                "passed": False,
                "rhat_threshold": config.verification_rhat_max,
                "diagnostic_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        payload["modern_rank_normalized_verification"] = modern
        payload["modern_verification_coordinate_system"] = config.verification_coordinate_system
    if require_efficiency:
        parameter_names = getattr(adapter.base_adapter, "parameter_names", None)
        if callable(parameter_names):
            parameter_names = parameter_names()
        if parameter_names is None:
            parameter_names = tuple(
                f"parameter[{index}]" for index in range(adapter.parameter_dim)
            )
        names = tuple(str(name) for name in parameter_names)
        if len(names) != adapter.parameter_dim:
            names = tuple(f"parameter[{index}]" for index in range(adapter.parameter_dim))
        payload["selection_efficiency_diagnostics"] = rank_normalized_hmc_diagnostics(
            raw,
            parameter_names=names,
            thresholds=RankNormalizedHMCThresholds(
                rhat_max=1.0e12,
                bulk_ess_min=1.0e-12,
                tail_ess_min=1.0e-12,
            ),
        )
    return payload


def _classify_verification(
    config: FixedTransportHMCKernelTuningConfig,
    *,
    diagnostics: Mapping[str, Any],
    run_error: Exception | None,
    require_modern: bool,
    require_efficiency: bool,
    require_complete_transition_telemetry: bool,
    acceptance_band: tuple[float, float],
    diagnostic_role: str,
) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    hard = []
    if run_error is not None:
        hard.append("verification_runtime_error")
    acceptance = _scalar_or_none(diagnostics.get("acceptance_rate"))
    if acceptance is None or not math.isfinite(acceptance):
        hard.append("verification_acceptance_missing_or_nonfinite")
    elif not acceptance_band[0] <= acceptance <= acceptance_band[1]:
        hard.append("verification_acceptance_outside_pass_band")
    hard.extend(
        _basic_hard_vetoes(
            diagnostics,
            prefix="verification",
            require_complete_transition_telemetry=(
                require_complete_transition_telemetry
            ),
        )
    )
    if config.require_all_chain_movement and diagnostics.get("all_chains_moved") is not True:
        hard.append("verification_chain_without_movement")
    divergence = _int_or_none(diagnostics.get("divergence_count"))
    if diagnostics.get("divergence_status") == "available" and divergence is not None and divergence > 0:
        hard.append("verification_native_divergence_detected")
    telemetry = diagnostics.get("target_status_telemetry")
    if config.target_status_trace_policy != "none":
        if not isinstance(telemetry, Mapping):
            hard.append("verification_target_status_telemetry_missing")
        elif bool(telemetry.get("telemetry_failure_veto")):
            hard.append("verification_target_status_telemetry_failure")
    if require_modern:
        modern = diagnostics.get("modern_rank_normalized_verification")
        if not isinstance(modern, Mapping) or modern.get("passed") is not True:
            hard.append("verification_modern_rank_folded_rhat_failed")
    if require_efficiency:
        efficiency = diagnostics.get("selection_efficiency_diagnostics")
        if (
            not isinstance(efficiency, Mapping)
            or efficiency.get("input_all_finite") is not True
            or efficiency.get("diagnostics_all_finite") is not True
        ):
            hard.append("verification_selection_efficiency_nonfinite")
    hard = list(dict.fromkeys(hard))
    repairs = []
    if acceptance is not None and math.isfinite(acceptance):
        if acceptance < acceptance_band[0]:
            repairs.append("verification_acceptance_below_pass_band")
        elif acceptance > acceptance_band[1]:
            repairs.append("verification_acceptance_above_pass_band")
    return (
        "passed" if not hard else "hard_veto",
        diagnostic_role if not hard else "hard_veto",
        tuple(hard),
        tuple(repairs),
    )


def _basic_hard_vetoes(
    diagnostics: Mapping[str, Any],
    *,
    prefix: str,
    require_complete_transition_telemetry: bool = False,
) -> list[str]:
    hard = []
    for key, reason in (
        ("samples_all_finite", "samples_nonfinite_or_missing"),
        ("log_accept_ratio_finite", "log_accept_nonfinite_or_missing"),
        ("target_log_prob_finite", "target_log_prob_nonfinite_or_missing"),
    ):
        if diagnostics.get(key) is not True:
            hard.append(f"{prefix}_{reason}")
    for key, reason in (
        ("proposed_target_log_prob_finite", "proposed_target_log_prob"),
        ("target_score_finite", "target_score"),
    ):
        status = diagnostics.get(key)
        if status is False:
            hard.append(f"{prefix}_{reason}_nonfinite")
        elif require_complete_transition_telemetry and status is not True:
            hard.append(f"{prefix}_{reason}_missing")
    return hard


def _chain_config(
    config: FixedTransportHMCKernelTuningConfig,
    *,
    num_results: int,
    burnin: int,
    step: float,
    leapfrog: int,
    seed: tuple[int, int],
    adaptive_steps: int,
) -> _FullChainHMCConfig:
    policy = (
        _TuningPolicy.fixed(source=config.source)
        if adaptive_steps == 0
        else _TuningPolicy.dual_averaging(
            steps=adaptive_steps,
            target=config.target_accept_prob,
            source=config.source,
        )
    )
    return _FullChainHMCConfig(
        num_results=int(num_results),
        num_burnin_steps=int(burnin),
        step_size=float(step),
        num_leapfrog_steps=int(leapfrog),
        seed=seed,
        use_xla=config.use_xla,
        trace_policy="standard",
        target_status_trace_policy=config.target_status_trace_policy,
        tuning_policy=policy,
        target_scope=str(config.target_scope or "fixed_transport_target"),
        chain_execution_mode=config.chain_execution_mode,
    )


def _initial_state(
    config: FixedTransportHMCKernelTuningConfig,
    parameter_dim: int,
    *,
    z0: tf.Tensor | None = None,
) -> tf.Tensor:
    if config.initial_state_bank:
        state = tf.convert_to_tensor(config.initial_state_bank, tf.float64)
        if state.shape != (config.chain_count, int(parameter_dim)):
            raise ValueError(
                "initial_state_bank must have shape [chain_count, parameter_dim]"
            )
        return state
    if z0 is not None:
        position = _validate_initial_position(z0, parameter_dim)
        return tf.broadcast_to(
            position[tf.newaxis, :],
            (config.chain_count, int(parameter_dim)),
        )
    return tf.zeros((config.chain_count, int(parameter_dim)), tf.float64)


def _validate_initial_position(value: Any, parameter_dim: int) -> tf.Tensor:
    tensor = tf.cast(tf.convert_to_tensor(value), tf.float64)
    if tensor.shape != (int(parameter_dim),):
        raise ValueError("initial_position must have shape [parameter_dim]")
    if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
        raise ValueError("initial_position must be finite")
    return tensor


def _identity_mass_payload(position: tf.Tensor, adapter_signature: str) -> Mapping[str, Any]:
    dimension = int(position.shape[0])
    identity = tf.eye(dimension, dtype=tf.float64)
    return {
        "artifact_type": "bayesfilter_precomputed_mass_artifact",
        "schema_version": 2,
        "include_arrays": True,
        "dimension": dimension,
        "adapter_signature": adapter_signature,
        "position": _json_ready(position),
        "covariance": _json_ready(identity),
        "factor": _json_ready(identity),
        "position_role": "fixed_neutra_initial_z",
        "covariance_source": "fixed_identity_z",
        "matrix_used_for_square_root": "identity_z",
        "factor_orientation": "row_right_transpose",
        "source": "bayesfilter.fixed_transport_hmc_tuning.identity_z_mass",
        "regularization_report": {"regularization_applied": False},
        "nonclaims": (
            "fixed identity mass in trained-transport z coordinates",
            "no residual mass adaptation claim",
        ),
    }


def _map_samples(adapter: FixedTransportValueScoreAdapter, samples: tf.Tensor) -> tf.Tensor:
    if samples.shape.rank != 3 or any(value is None for value in samples.shape):
        raise ValueError("verification samples must have static [draw, chain, parameter] shape")
    draws, chains, parameters = (int(value) for value in samples.shape)
    flat = tf.reshape(samples, (draws * chains, parameters))
    mapped = tf.cast(adapter.latent_to_position(flat), tf.float64)
    if mapped.shape != flat.shape:
        raise ValueError("fixed transport changed verification sample shape")
    return tf.reshape(mapped, (draws, chains, parameters))


def _select_candidate(
    candidates: Sequence[FixedTransportHMCCandidateResult],
    config: FixedTransportHMCKernelTuningConfig,
) -> int | None:
    viable = [(index, candidate) for index, candidate in enumerate(candidates) if candidate.passed]
    if not viable:
        return None
    target = config.target_accept_prob
    return min(
        viable,
        key=lambda item: (
            abs(float(item[1].selected_acceptance_rate) - target),
            item[1].num_leapfrog_steps,
            float(item[1].selected_step_size),
            item[0],
        ),
    )[0]


def _select_candidates(
    candidates: Sequence[FixedTransportHMCCandidateResult],
    config: FixedTransportHMCKernelTuningConfig,
    *,
    adapter: FixedTransportValueScoreAdapter,
    z0: tf.Tensor,
    run_full_chain: RunFullChainFn,
    passthrough_exceptions: tuple[type[Exception], ...],
) -> Mapping[str, Any]:
    """Select among candidate artifacts and qualify exactly one winner.

    The efficiency policy runs its discarded, fixed-kernel replications once
    per ladder nominee. It never uses one candidate's screen to infer another
    candidate's efficiency and never reuses selection or held-out draws as
    posterior draws.
    """

    candidate_rows: list[dict[str, Any]] = []
    selection_hard_vetoes: list[str] = []
    selection_repair_triggers: list[str] = []
    for index, candidate in enumerate(candidates):
        row: dict[str, Any] = {
            "candidate_index": index,
            "num_leapfrog_steps": candidate.num_leapfrog_steps,
            "step_size": candidate.selected_step_size,
            "ladder_or_compatibility_verification_passed": candidate.passed,
            "independent_verification_passed": (
                candidate.passed
                if config.selection_policy == "acceptance_target_distance"
                else None
            ),
            "acceptance_rate": candidate.selected_acceptance_rate,
            "candidate_artifact_hash": candidate.artifact_hash,
            "hard_vetoes": candidate.hard_vetoes,
            "repair_triggers": candidate.repair_triggers,
        }
        if (
            config.selection_policy == "replicated_min_bulk_ess_per_gradient"
            and candidate.passed
        ):
            if candidate.selected_step_size is None:
                raise RuntimeError("passed candidate has no selected step size")
            evidence = _run_replicated_efficiency_selection(
                config,
                adapter=adapter,
                z0=z0,
                step=float(candidate.selected_step_size),
                leapfrog=candidate.num_leapfrog_steps,
                candidate_index=index,
                run_full_chain=run_full_chain,
                passthrough_exceptions=passthrough_exceptions,
            )
            diagnostics = evidence["diagnostics"]
            eligible = evidence["final_status"] == "passed"
            row.update(
                {
                    "selection_evidence": evidence,
                    "eligible": eligible,
                    "minimum_bulk_ess_per_declared_target_gradient": (
                        diagnostics.get(
                            "minimum_bulk_ess_per_declared_target_gradient"
                        )
                    ),
                    "maximum_modern_rhat_across_replications": diagnostics.get(
                        "maximum_modern_rhat_across_replications"
                    ),
                }
            )
            selection_hard_vetoes.extend(
                f"candidate_{index}_{reason}" for reason in evidence["hard_vetoes"]
            )
            selection_repair_triggers.extend(
                f"candidate_{index}_{reason}"
                for reason in evidence["repair_triggers"]
            )
        else:
            row.update(
                {
                    "selection_evidence": None,
                    "eligible": candidate.passed,
                    "minimum_bulk_ess_per_declared_target_gradient": None,
                    "maximum_modern_rhat_across_replications": None,
                }
            )
        candidate_rows.append(row)

    if config.selection_policy == "replicated_min_bulk_ess_per_gradient":
        viable_rows = [row for row in candidate_rows if row["eligible"]]
        selected = (
            None
            if not viable_rows
            else int(
                min(
                    viable_rows,
                    key=lambda row: (
                        -float(
                            row[
                                "minimum_bulk_ess_per_declared_target_gradient"
                            ]
                        ),
                        float(row["maximum_modern_rhat_across_replications"]),
                        int(row["num_leapfrog_steps"]),
                        float(row["step_size"]),
                        int(row["candidate_index"]),
                    ),
                )["candidate_index"]
            )
        )
    else:
        selected = _select_candidate(candidates, config)

    nominated = selected
    heldout = None
    if (
        config.selection_policy == "replicated_min_bulk_ess_per_gradient"
        and nominated is not None
    ):
        nominated_candidate = candidates[nominated]
        if nominated_candidate.selected_step_size is None:
            raise RuntimeError("nominated candidate has no selected step size")
        heldout = _run_verification(
            config,
            adapter=adapter,
            z0=z0,
            step=float(nominated_candidate.selected_step_size),
            leapfrog=nominated_candidate.num_leapfrog_steps,
            candidate_index=nominated,
            run_full_chain=run_full_chain,
            probe_only=False,
            passthrough_exceptions=passthrough_exceptions,
            post_selection_heldout=True,
        )
        if heldout["final_status"] != "passed":
            selected = None
            selection_hard_vetoes.extend(
                f"heldout_{reason}" for reason in heldout["hard_vetoes"]
            )
            selection_repair_triggers.extend(
                f"heldout_{reason}" for reason in heldout["repair_triggers"]
            )

    independently_viable = any(candidate.passed for candidate in candidates)
    if nominated is None:
        final_status = (
            "no_viable_selection_candidate"
            if independently_viable
            else "no_viable_candidate"
        )
        selection_hard_vetoes.append(final_status)
    elif selected is None:
        final_status = "heldout_verification_failed"
    else:
        final_status = "passed"
    seed_ledger = _selection_seed_ledger(
        candidates=candidates,
        candidate_rows=candidate_rows,
        heldout=heldout,
        nominated_candidate_index=nominated,
    )
    return {
        "schema": "bayesfilter.fixed_transport_hmc_candidate_selection.v3",
        "selection_policy": config.selection_policy,
        "selection_acceptance_band": config.selection_acceptance_band,
        "selection_replications": config.selection_replications,
        "selection_num_results": config.selection_num_results,
        "selection_num_burnin_steps": config.selection_num_burnin_steps,
        "transformed_adapter_signature": adapter.adapter_signature(),
        "target_scope": config.target_scope,
        "candidate_rows": tuple(candidate_rows),
        "nominated_candidate_index": nominated,
        "selected_candidate_index": selected,
        "heldout_verification": heldout,
        "candidate_verification_serves_as_final": (
            config.selection_policy == "acceptance_target_distance"
        ),
        "post_selection_candidate_only_verification_used": (
            config.selection_policy == "replicated_min_bulk_ess_per_gradient"
            and nominated is not None
        ),
        "seed_ledger": seed_ledger,
        "selection_order": (
            "maximize minimum bulk ESS per declared target gradient; "
            "minimize maximum modern R-hat; L; step size; candidate order"
            if config.selection_policy == "replicated_min_bulk_ess_per_gradient"
            else "minimize acceptance target distance; L; step size; candidate order"
        ),
        "final_status": final_status,
        "hard_vetoes": tuple(dict.fromkeys(selection_hard_vetoes)),
        "repair_triggers": tuple(dict.fromkeys(selection_repair_triggers)),
        "all_selection_draws_discarded": True,
        "all_heldout_draws_discarded": heldout is not None,
        "reports_posterior_convergence": False,
        "reports_sampler_superiority": False,
    }


def _selection_seed_ledger(
    *,
    candidates: Sequence[FixedTransportHMCCandidateResult],
    candidate_rows: Sequence[Mapping[str, Any]],
    heldout: Mapping[str, Any] | None,
    nominated_candidate_index: int | None,
) -> Mapping[str, Any]:
    """Record and validate disjoint adaptation, selection, and holdout seeds."""

    rows: list[dict[str, Any]] = []

    def append(role: str, candidate_index: int, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            return
        seed = payload.get("seed")
        if not isinstance(seed, (tuple, list)) or len(seed) != 2:
            raise RuntimeError(f"{role} config lacks a two-integer seed")
        rows.append(
            {
                "role": role,
                "candidate_index": int(candidate_index),
                "seed": tuple(int(value) for value in seed),
            }
        )

    for candidate in candidates:
        ladder = candidate.ladder_result
        if isinstance(ladder, Mapping):
            for round_payload in ladder.get("rounds", ()):
                if not isinstance(round_payload, Mapping):
                    continue
                append("dual_averaging", candidate.candidate_index, round_payload.get("tune_config"))
                append("ladder_screen", candidate.candidate_index, round_payload.get("screen_config"))
        if (
            candidate.diagnostic_role == "fresh_fixed_kernel_candidate_verification"
            or candidate.ladder_result is None
        ):
            append(
                "candidate_verification_or_fixed_grid_screen",
                candidate.candidate_index,
                candidate.verification_config_payload,
            )

    for row in candidate_rows:
        evidence = row.get("selection_evidence")
        if not isinstance(evidence, Mapping):
            continue
        configs = evidence.get("config_payload", {}).get("replications", ())
        for replication, payload in enumerate(configs):
            append(
                f"selection_replication_{replication}",
                int(row["candidate_index"]),
                payload,
            )
    if isinstance(heldout, Mapping):
        if nominated_candidate_index is None:
            raise RuntimeError("held-out verification lacks a nominated candidate")
        append(
            "post_selection_heldout",
            nominated_candidate_index,
            heldout.get("config_payload"),
        )

    seeds = tuple(row["seed"] for row in rows)
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("fixed-transport tuning seed domains overlap")
    return {
        "schema": "bayesfilter.fixed_transport_hmc_seed_ledger.v1",
        "rows": tuple(rows),
        "seed_count": len(rows),
        "all_seeds_unique": True,
    }


def _acceptance_class(
    acceptance: float | None, config: FixedTransportHMCKernelTuningConfig
) -> str:
    if acceptance is None or not math.isfinite(acceptance):
        return "invalid"
    if config.acceptance_band[0] <= acceptance <= config.acceptance_band[1]:
        return "in_band"
    if acceptance < config.acceptance_band[0]:
        return "below_band"
    if acceptance <= config.fixed_grid_fallback_acceptance_max:
        return "high_warning_band"
    return "above_fallback_band"


def _diagnostic_roles() -> Mapping[str, str]:
    return {
        "base_value_score_authority": "hard_veto_if_gradient_tape_fallback",
        "fixed_transport_manifest_hash": "artifact_identity",
        "identity_z_mass_policy": "hard_boundary",
        "candidate_ladder": "step_tuning_screen",
        "fresh_verification": "handoff_promotion_screen",
        "all_chain_movement": "hard_veto_in_every_fixed_kernel_screen",
        "replicated_efficiency_selection": "discarded_candidate_nomination_screen",
        "tuning_scope": "handoff_lineage_boundary",
        "source_dependency_closure": "executable_provenance_boundary",
        "acceptance": "promotion_screen_and_repair_trigger",
        "native_divergence": "hard_veto_only_when_available_and_positive",
        "native_divergence_unavailable": "recorded_nonclaim_not_veto_not_zero",
        "max_abs_log_accept_energy_proxy": "explanatory_alert_only",
        "modern_rank_folded_rhat": "required_when_configured_handoff_promotion_screen",
        "runtime": "explanatory_diagnostic",
    }


def _error_diagnostics(error: Exception) -> Mapping[str, Any]:
    return {
        "runtime_error_type": type(error).__name__,
        "runtime_error_message": str(error),
        "samples_all_finite": False,
        "log_accept_ratio_finite": False,
        "target_log_prob_finite": False,
        "proposed_target_log_prob_finite": None,
        "target_score_finite": None,
        "acceptance_rate": None,
        "divergence_status": "not_collected",
        "divergence_count": None,
        "modern_rank_normalized_verification": None,
    }


def _base_adapter_signature(adapter: Any) -> str:
    explicit = getattr(adapter, "adapter_signature", None)
    if explicit is not None:
        return str(explicit() if callable(explicit) else explicit)
    return _stable_hash(
        {
            "module": adapter.__class__.__module__,
            "class": adapter.__class__.__qualname__,
            "parameter_dim": int(getattr(adapter, "parameter_dim")),
        }
    )


def _tuning_source_dependency_closure() -> Mapping[str, Any]:
    """Hash the executable source boundary used by this tuning artifact."""

    inference_root = Path(__file__).resolve().parent
    package_root = inference_root.parent
    paths = (
        inference_root / "fixed_transport_hmc_tuning_tf.py",
        inference_root / "fixed_transport_hmc_tuning.py",
        inference_root / "fixed_transport_hmc_mechanics_tf.py",
        inference_root / "batched_value_score.py",
        inference_root / "hmc_convergence.py",
        inference_root / "tuning_contract.py",
        inference_root / "posterior_adapter.py",
        inference_root / "hmc.py",
        inference_root / "__init__.py",
        package_root / "__init__.py",
    )
    missing = tuple(str(path) for path in paths if not path.is_file())
    if missing:
        raise RuntimeError(
            "fixed-transport tuning source closure is incomplete: "
            + ", ".join(missing)
        )
    return {
        "schema": "bayesfilter.fixed_transport_hmc_source_closure.v1",
        "files": tuple(
            {
                "path": str(path.relative_to(package_root.parent)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ),
    }


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    if tf.is_tensor(value):
        return _json_ready(value.numpy())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _scalar_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if tf.is_tensor(value):
        tensor = tf.reshape(tf.convert_to_tensor(value), (-1,))
        if int(tf.size(tensor).numpy()) == 0:
            return None
        return float(tensor[-1].numpy())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    scalar = _scalar_or_none(value)
    return None if scalar is None else int(scalar)


def _positive_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _validate_band(values: Sequence[float], *, name: str) -> tuple[float, float]:
    raw = tuple(float(value) for value in values)
    if len(raw) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower, upper = raw
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"{name} values must be finite")
    if not 0.0 < lower <= upper < 1.0:
        raise ValueError(f"{name} must satisfy 0 < lower <= upper < 1")
    return lower, upper


def _validate_seed(value: Sequence[int]) -> tuple[int, int]:
    seed = tuple(int(item) for item in value)
    if len(seed) != 2:
        raise ValueError("seed must contain exactly two integers")
    return seed


def _derived_seed(seed: tuple[int, int], *components: int) -> tuple[int, int]:
    """Fold phase and candidate coordinates into a stateless seed.

    Additive offsets can collide across phases (for example, candidate one in
    adaptation and candidate zero in screening). Stateless fold-in gives every
    phase/candidate/round tuple a separate deterministic seed domain.
    """

    derived = tf.convert_to_tensor(seed, dtype=tf.int32)
    for component in components:
        derived = tf.random.experimental.stateless_fold_in(
            derived, tf.convert_to_tensor(int(component), dtype=tf.int32)
        )
    values = tuple(int(value) for value in derived.numpy().tolist())
    if len(values) != 2:
        raise RuntimeError("derived stateless seed did not have width two")
    return values


def _string_tuple(value: Sequence[str] | str) -> tuple[str, ...]:
    values = (value,) if isinstance(value, str) else tuple(value)
    return tuple(str(item) for item in values if str(item))


__all__ = [
    "FIXED_TRANSPORT_HMC_TUNING_NONCLAIMS",
    "FixedTransportHMCCandidateResult",
    "FixedTransportHMCKernelTuningConfig",
    "FixedTransportHMCKernelTuningResult",
    "VerifiedFixedTransportHMCHandoff",
    "build_verified_fixed_transport_hmc_handoff_from_tuning_result",
    "tune_fixed_transport_hmc_kernel",
]
