"""Fixed-transport HMC tuning for trained NeuTra-style maps.

This module owns the BayesFilter policy for tuning an HMC kernel after a
nonlinear transport has already been trained.  The transport is fixed input
geometry; this tuner does not adapt mass and does not call the windowed-mass
kernel tuner.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
from bayesfilter.inference.hmc import (
    FullChainHMCConfig,
    FullChainHMCRunResult,
    PrecomputedMassArtifact,
    run_full_chain_tfp_hmc,
    stable_adapter_signature,
)
from bayesfilter.inference.hmc_convergence import (
    RANK_NORMALIZED_SPLIT_RHAT_DEFINITION,
    rank_normalized_split_rhat_summary,
)
from bayesfilter.inference.hmc_budget_ladder import (
    FixedMassHMCTuningBudgetLadderConfig,
    FixedMassHMCTuningBudgetLadderResult,
    run_fixed_mass_hmc_tuning_budget_ladder,
)
from bayesfilter.inference.generic_hmc_tuning import (
    classify_hmc_fixed_grid_acceptance,
    select_hmc_fixed_grid_scale,
)
from bayesfilter.inference.posterior_adapter import value_score_capability
from bayesfilter.runtime import stable_config_hash


FIXED_TRANSPORT_HMC_TUNING_NONCLAIMS: tuple[str, ...] = (
    "fixed trained transport HMC tuning only",
    "transport is not trained or adapted by this tuner",
    "identity z-mass policy only",
    "no windowed mass adaptation claim",
    "no posterior convergence claim",
    "no sampler superiority claim",
    "no default-readiness claim",
    "no external-client scientific claim",
)

_FORBIDDEN_BASE_AUTHORITIES = frozenset({"gradient_tape_fallback"})


RunFullChainFn = Callable[[Any, Any, FullChainHMCConfig], FullChainHMCRunResult]


@dataclass(frozen=True)
class FixedTransportHMCKernelTuningConfig:
    """Policy for fixed-NeuTra HMC kernel tuning in transport coordinates."""

    initial_step_size: float
    leapfrog_grid: tuple[int, ...] = (5, 10, 15, 20, 25)
    chain_count: int = 4
    target_accept_prob: float = 0.70
    acceptance_band: tuple[float, float] = (0.65, 0.75)
    repair_band: tuple[float, float] = (0.55, 0.85)
    maximum_absolute_energy_error: float = 1000.0
    step_repair_factor: float = 2.0
    step_repair_min_directional_factor: float = 1.25
    budget_schedule: tuple[int, ...] = (8, 16, 32)
    tune_num_results: int = 8
    screen_num_results: int = 16
    screen_num_burnin_steps: int = 4
    verification_num_results: int = 16
    verification_num_burnin_steps: int = 4
    require_modern_rank_normalized_verification: bool = False
    verification_min_retained_results_per_chain: int = 1000
    verification_rhat_max: float = 1.01
    tune_seed_base: tuple[int, int] = (20260625, 100)
    screen_seed_base: tuple[int, int] = (20260625, 200)
    verification_seed_base: tuple[int, int] = (20260625, 300)
    chain_execution_mode: str = "tf_function"
    use_xla: bool = False
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
        step = float(self.initial_step_size)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("initial_step_size must be positive and finite")
        object.__setattr__(self, "initial_step_size", step)
        grid = tuple(int(item) for item in self.leapfrog_grid)
        if not grid or any(item <= 0 for item in grid):
            raise ValueError("leapfrog_grid must contain positive integers")
        object.__setattr__(self, "leapfrog_grid", tuple(dict.fromkeys(grid)))
        chain_count = int(self.chain_count)
        if chain_count <= 0:
            raise ValueError("chain_count must be positive")
        object.__setattr__(self, "chain_count", chain_count)
        target = float(self.target_accept_prob)
        if not np.isfinite(target) or not 0.0 < target < 1.0:
            raise ValueError("target_accept_prob must be finite and in (0, 1)")
        object.__setattr__(self, "target_accept_prob", target)
        acceptance_band = _validate_band(self.acceptance_band, name="acceptance_band")
        repair_band = _validate_band(self.repair_band, name="repair_band")
        if repair_band[0] > acceptance_band[0] or repair_band[1] < acceptance_band[1]:
            raise ValueError("repair_band must contain acceptance_band")
        object.__setattr__(self, "acceptance_band", acceptance_band)
        object.__setattr__(self, "repair_band", repair_band)
        dynamics_identity = str(self.proposal_dynamics_identity)
        if not dynamics_identity:
            raise ValueError("proposal_dynamics_identity must be non-empty")
        object.__setattr__(self, "proposal_dynamics_identity", dynamics_identity)
        maximum_energy = float(self.maximum_absolute_energy_error)
        if not np.isfinite(maximum_energy) or maximum_energy <= 0.0:
            raise ValueError("maximum_absolute_energy_error must be positive and finite")
        object.__setattr__(self, "maximum_absolute_energy_error", maximum_energy)
        object.__setattr__(
            self,
            "step_repair_factor",
            _validate_step_repair_multiplier(
                self.step_repair_factor,
                name="step_repair_factor",
            ),
        )
        object.__setattr__(
            self,
            "step_repair_min_directional_factor",
            _validate_step_repair_multiplier(
                self.step_repair_min_directional_factor,
                name="step_repair_min_directional_factor",
            ),
        )
        budgets = tuple(int(item) for item in self.budget_schedule)
        if not budgets or any(item <= 0 for item in budgets):
            raise ValueError("budget_schedule must contain positive integers")
        object.__setattr__(self, "budget_schedule", budgets)
        for name in (
            "tune_num_results",
            "screen_num_results",
            "screen_num_burnin_steps",
            "verification_num_results",
            "verification_num_burnin_steps",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        require_modern = bool(self.require_modern_rank_normalized_verification)
        minimum_verification = int(self.verification_min_retained_results_per_chain)
        if minimum_verification <= 0:
            raise ValueError(
                "verification_min_retained_results_per_chain must be positive"
            )
        verification_rhat_max = float(self.verification_rhat_max)
        if not np.isfinite(verification_rhat_max) or verification_rhat_max <= 1.0:
            raise ValueError("verification_rhat_max must be finite and greater than 1")
        if require_modern:
            if chain_count != 4:
                raise ValueError(
                    "modern rank-normalized verification requires exactly four chains"
                )
            if int(self.verification_num_results) < minimum_verification:
                raise ValueError(
                    "modern rank-normalized verification requires at least the "
                    "configured retained results per chain"
                )
        object.__setattr__(
            self,
            "require_modern_rank_normalized_verification",
            require_modern,
        )
        object.__setattr__(
            self,
            "verification_min_retained_results_per_chain",
            minimum_verification,
        )
        object.__setattr__(self, "verification_rhat_max", verification_rhat_max)
        object.__setattr__(self, "tune_seed_base", _validate_seed(self.tune_seed_base))
        screen_seed = _validate_seed(self.screen_seed_base)
        verification_seed = _validate_seed(self.verification_seed_base)
        if screen_seed == self.tune_seed_base:
            raise ValueError("screen_seed_base must differ from tune_seed_base")
        if verification_seed in {self.tune_seed_base, screen_seed}:
            raise ValueError("verification_seed_base must be distinct")
        object.__setattr__(self, "screen_seed_base", screen_seed)
        object.__setattr__(self, "verification_seed_base", verification_seed)
        mode = str(self.chain_execution_mode)
        if mode not in {"tf_function", "eager"}:
            raise ValueError("chain_execution_mode must be 'tf_function' or 'eager'")
        if self.use_xla and mode != "tf_function":
            raise ValueError("XLA HMC requires chain_execution_mode='tf_function'")
        object.__setattr__(self, "chain_execution_mode", mode)
        object.__setattr__(self, "use_xla", bool(self.use_xla))
        if self.target_scope is not None:
            scope = str(self.target_scope)
            if not scope:
                raise ValueError("target_scope must be non-empty when provided")
            object.__setattr__(self, "target_scope", scope)
        status_policy = str(self.target_status_trace_policy)
        if status_policy not in {"none", "per_chain_step"}:
            raise ValueError(
                "target_status_trace_policy must be 'none' or 'per_chain_step'"
            )
        object.__setattr__(self, "target_status_trace_policy", status_policy)
        fixed_grid_base = tuple(
            float(item) for item in self.fixed_grid_base_step_size_candidates
        )
        if any((not np.isfinite(item)) or item <= 0.0 for item in fixed_grid_base):
            raise ValueError(
                "fixed_grid_base_step_size_candidates must be positive and finite"
            )
        fixed_grid_scales = tuple(float(item) for item in self.fixed_grid_scale_candidates)
        if any((not np.isfinite(item)) or item <= 0.0 for item in fixed_grid_scales):
            raise ValueError("fixed_grid_scale_candidates must be positive and finite")
        if fixed_grid_scales and tuple(sorted(fixed_grid_scales)) != fixed_grid_scales:
            raise ValueError("fixed_grid_scale_candidates must be sorted ascending")
        if bool(fixed_grid_base) != bool(fixed_grid_scales):
            raise ValueError(
                "fixed_grid_base_step_size_candidates and fixed_grid_scale_candidates "
                "must be provided together"
            )
        object.__setattr__(
            self,
            "fixed_grid_base_step_size_candidates",
            fixed_grid_base,
        )
        object.__setattr__(self, "fixed_grid_scale_candidates", fixed_grid_scales)
        if self.fixed_grid_num_leapfrog_steps is not None:
            fixed_grid_leapfrog = int(self.fixed_grid_num_leapfrog_steps)
            if fixed_grid_leapfrog <= 0:
                raise ValueError("fixed_grid_num_leapfrog_steps must be positive")
            object.__setattr__(
                self,
                "fixed_grid_num_leapfrog_steps",
                fixed_grid_leapfrog,
            )
        fixed_grid_attempts = int(self.fixed_grid_max_attempts)
        if fixed_grid_attempts <= 0:
            raise ValueError("fixed_grid_max_attempts must be positive")
        object.__setattr__(self, "fixed_grid_max_attempts", fixed_grid_attempts)
        fallback = float(self.fixed_grid_fallback_acceptance_max)
        if not np.isfinite(fallback) or fallback < acceptance_band[1] or fallback >= 1.0:
            raise ValueError(
                "fixed_grid_fallback_acceptance_max must be finite and in "
                "[acceptance_band upper, 1)"
            )
        object.__setattr__(self, "fixed_grid_fallback_acceptance_max", fallback)
        filename = str(self.output_filename)
        if not filename or Path(filename).name != filename:
            raise ValueError("output_filename must be a plain file name")
        object.__setattr__(self, "output_filename", filename)
        source = str(self.source)
        if not source:
            raise ValueError("source must be non-empty")
        object.__setattr__(self, "source", source)

    def payload(self) -> Mapping[str, Any]:
        return {
            "initial_step_size": self.initial_step_size,
            "leapfrog_grid": self.leapfrog_grid,
            "chain_count": self.chain_count,
            "target_accept_prob": self.target_accept_prob,
            "acceptance_band": self.acceptance_band,
            "repair_band": self.repair_band,
            "maximum_absolute_energy_error": self.maximum_absolute_energy_error,
            "proposal_dynamics_identity": self.proposal_dynamics_identity,
            "step_repair_factor": self.step_repair_factor,
            "step_repair_min_directional_factor": (
                self.step_repair_min_directional_factor
            ),
            "budget_schedule": self.budget_schedule,
            "tune_num_results": self.tune_num_results,
            "screen_num_results": self.screen_num_results,
            "screen_num_burnin_steps": self.screen_num_burnin_steps,
            "verification_num_results": self.verification_num_results,
            "verification_num_burnin_steps": self.verification_num_burnin_steps,
            "require_modern_rank_normalized_verification": (
                self.require_modern_rank_normalized_verification
            ),
            "verification_min_retained_results_per_chain": (
                self.verification_min_retained_results_per_chain
            ),
            "verification_rhat_max": self.verification_rhat_max,
            "tune_seed_base": self.tune_seed_base,
            "screen_seed_base": self.screen_seed_base,
            "verification_seed_base": self.verification_seed_base,
            "chain_execution_mode": self.chain_execution_mode,
            "use_xla": self.use_xla,
            "target_scope": self.target_scope,
            "target_status_trace_policy": self.target_status_trace_policy,
            "fixed_grid_base_step_size_candidates": (
                self.fixed_grid_base_step_size_candidates
            ),
            "fixed_grid_scale_candidates": self.fixed_grid_scale_candidates,
            "fixed_grid_num_leapfrog_steps": self.fixed_grid_num_leapfrog_steps,
            "fixed_grid_max_attempts": self.fixed_grid_max_attempts,
            "fixed_grid_fallback_acceptance_max": (
                self.fixed_grid_fallback_acceptance_max
            ),
            "output_filename": self.output_filename,
            "source": self.source,
        }


@dataclass(frozen=True)
class FixedTransportHMCCandidateResult:
    """One fixed-L candidate in the fixed-transport tuning grid."""

    candidate_index: int
    num_leapfrog_steps: int
    ladder_result: FixedMassHMCTuningBudgetLadderResult | None
    verification_config_payload: Mapping[str, Any] | None
    verification_diagnostics: Mapping[str, Any]
    final_status: str
    diagnostic_role: str
    fixed_kernel_step_size: float | None = None
    hard_vetoes: tuple[str, ...] = ()
    repair_triggers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_index", int(self.candidate_index))
        leapfrog = int(self.num_leapfrog_steps)
        if leapfrog <= 0:
            raise ValueError("num_leapfrog_steps must be positive")
        object.__setattr__(self, "num_leapfrog_steps", leapfrog)
        object.__setattr__(
            self,
            "verification_config_payload",
            None
            if self.verification_config_payload is None
            else dict(self.verification_config_payload),
        )
        object.__setattr__(self, "verification_diagnostics", dict(self.verification_diagnostics))
        if self.ladder_result is None and self.fixed_kernel_step_size is None:
            raise ValueError("candidate requires ladder_result or fixed_kernel_step_size")
        fixed_step = (
            None
            if self.fixed_kernel_step_size is None
            else float(self.fixed_kernel_step_size)
        )
        if fixed_step is not None and (not np.isfinite(fixed_step) or fixed_step <= 0.0):
            raise ValueError("fixed_kernel_step_size must be positive and finite")
        object.__setattr__(self, "fixed_kernel_step_size", fixed_step)
        object.__setattr__(self, "final_status", str(self.final_status))
        object.__setattr__(self, "diagnostic_role", str(self.diagnostic_role))
        object.__setattr__(self, "hard_vetoes", _string_tuple(self.hard_vetoes))
        object.__setattr__(self, "repair_triggers", _string_tuple(self.repair_triggers))

    @property
    def passed(self) -> bool:
        return self.final_status == "passed"

    @property
    def selected_step_size(self) -> float | None:
        if self.fixed_kernel_step_size is not None:
            return float(self.fixed_kernel_step_size)
        if self.ladder_result is None:
            return None
        selected = self.ladder_result.selected_round
        if selected is None or selected.tuned_step_size is None:
            return None
        return float(selected.tuned_step_size)

    @property
    def selected_acceptance_rate(self) -> float | None:
        acceptance = self.verification_diagnostics.get("acceptance_rate")
        return _scalar_or_none(acceptance)

    @property
    def artifact_hash(self) -> str:
        return stable_config_hash(self.payload())

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
                None if self.ladder_result is None else self.ladder_result.artifact_hash
            ),
            "ladder": None if self.ladder_result is None else self.ladder_result.payload(),
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
    """Result and frozen-kernel handoff for fixed-transport HMC tuning."""

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
    artifact_path: str | None = None
    fixed_grid_scale_selection_payload: Mapping[str, Any] | None = None
    diagnostic_roles: Mapping[str, str] | None = None
    hard_vetoes: tuple[str, ...] = ()
    repair_triggers: tuple[str, ...] = ()
    nonclaims: tuple[str, ...] = FIXED_TRANSPORT_HMC_TUNING_NONCLAIMS

    def __post_init__(self) -> None:
        for name in (
            "transformed_adapter_signature",
            "base_adapter_signature",
            "fixed_transport_manifest_hash",
            "identity_z_mass_artifact_signature",
            "final_status",
        ):
            value = str(getattr(self, name))
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        dimension = int(self.target_dimension)
        if dimension <= 0:
            raise ValueError("target_dimension must be positive")
        object.__setattr__(self, "target_dimension", dimension)
        candidates = tuple(self.candidates)
        object.__setattr__(self, "candidates", candidates)
        selected = (
            None if self.selected_candidate_index is None else int(self.selected_candidate_index)
        )
        if selected is not None and selected not in range(len(candidates)):
            raise ValueError("selected_candidate_index must refer to a candidate")
        if selected is not None and not candidates[selected].passed:
            raise ValueError("selected candidate must have passed")
        object.__setattr__(self, "selected_candidate_index", selected)
        object.__setattr__(
            self,
            "final_kernel_payload",
            None if self.final_kernel_payload is None else dict(self.final_kernel_payload),
        )
        object.__setattr__(
            self,
            "identity_z_mass_artifact_payload",
            dict(self.identity_z_mass_artifact_payload),
        )
        object.__setattr__(
            self,
            "fixed_grid_scale_selection_payload",
            None
            if self.fixed_grid_scale_selection_payload is None
            else dict(self.fixed_grid_scale_selection_payload),
        )
        roles = {} if self.diagnostic_roles is None else dict(self.diagnostic_roles)
        object.__setattr__(self, "diagnostic_roles", roles)
        object.__setattr__(self, "hard_vetoes", _string_tuple(self.hard_vetoes))
        object.__setattr__(self, "repair_triggers", _string_tuple(self.repair_triggers))
        nonclaims = _string_tuple(self.nonclaims)
        if not nonclaims:
            raise ValueError("nonclaims must be non-empty")
        object.__setattr__(self, "nonclaims", nonclaims)

    @property
    def passed(self) -> bool:
        return self.selected_candidate_index is not None and self.final_kernel_payload is not None

    @property
    def selected_candidate(self) -> FixedTransportHMCCandidateResult | None:
        if self.selected_candidate_index is None:
            return None
        return self.candidates[self.selected_candidate_index]

    @property
    def final_kernel_hash(self) -> str | None:
        payload = self.final_kernel_payload
        return None if payload is None else stable_config_hash(payload)

    @property
    def artifact_hash(self) -> str:
        return stable_config_hash(self.payload())

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.fixed_transport_hmc_kernel_tuning_result.v1",
            "config": self.config.payload(),
            "transformed_adapter_signature": self.transformed_adapter_signature,
            "base_adapter_signature": self.base_adapter_signature,
            "fixed_transport_manifest_hash": self.fixed_transport_manifest_hash,
            "target_dimension": self.target_dimension,
            "identity_z_mass_artifact_payload": self.identity_z_mass_artifact_payload,
            "identity_z_mass_artifact_signature": self.identity_z_mass_artifact_signature,
            "candidates": tuple(candidate.payload() for candidate in self.candidates),
            "selected_candidate_index": self.selected_candidate_index,
            "final_status": self.final_status,
            "final_kernel_payload": self.final_kernel_payload,
            "final_kernel_hash": self.final_kernel_hash,
            "artifact_path": self.artifact_path,
            "fixed_grid_scale_selection": self.fixed_grid_scale_selection_payload,
            "diagnostic_roles": self.diagnostic_roles,
            "hard_vetoes": self.hard_vetoes,
            "repair_triggers": self.repair_triggers,
            "passed": self.passed,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "reports_default_readiness": False,
            "reports_external_client_scientific_claim": False,
            "nonclaims": self.nonclaims,
        }


def tune_fixed_transport_hmc_kernel(
    *,
    base_adapter: Any,
    fixed_transport: Any,
    initial_position: Any,
    config: FixedTransportHMCKernelTuningConfig | None = None,
    output_dir: str | Path | None = None,
    run_full_chain: RunFullChainFn = run_full_chain_tfp_hmc,
) -> FixedTransportHMCKernelTuningResult:
    """Tune a fixed-NeuTra HMC kernel without mass adaptation.

    The caller supplies a trained fixed transport and a reviewed base
    value/score adapter.  The resulting HMC target is the transformed density in
    z coordinates.  The mass policy is fixed identity in z; step size and
    leapfrog count are selected by a finite grid of fixed-mass dual-averaging
    screens followed by fresh fixed-kernel verification.
    """

    cfg = (
        FixedTransportHMCKernelTuningConfig(initial_step_size=0.1)
        if config is None
        else config
    )
    if not isinstance(cfg, FixedTransportHMCKernelTuningConfig):
        raise TypeError("config must be FixedTransportHMCKernelTuningConfig")
    _validate_base_adapter_authority(base_adapter)
    transformed_adapter = FixedTransportValueScoreAdapter(
        base_adapter=base_adapter,
        transport=fixed_transport,
        target_scope=_resolve_target_scope(base_adapter, cfg),
        batch_native=True,
        xla_hmc_ready=cfg.use_xla,
        full_chain_xla_diagnostic_ready=cfg.use_xla,
    )
    transformed_signature = stable_adapter_signature(transformed_adapter)
    base_signature = _base_adapter_signature(base_adapter)
    z0 = _validate_initial_position(initial_position, transformed_adapter.parameter_dim)
    identity_mass = _identity_z_mass_artifact(
        position=z0,
        adapter_signature=transformed_signature,
    )
    identity_mass_signature = _mass_artifact_signature(identity_mass)
    target_dimension = int(transformed_adapter.parameter_dim)
    candidates, scale_payload = _run_fixed_transport_candidate_attempts(
        cfg,
        transformed_adapter=transformed_adapter,
        identity_mass=identity_mass,
        initial_position=z0,
        target_dimension=target_dimension,
        run_full_chain=run_full_chain,
    )

    selected_index = _select_candidate(candidates, cfg)
    selected_candidate = None if selected_index is None else candidates[selected_index]
    final_kernel_payload = None
    final_status = "no_viable_candidate"
    hard_vetoes = tuple(
        dict.fromkeys(veto for candidate in candidates for veto in candidate.hard_vetoes)
    )
    repair_triggers = tuple(
        dict.fromkeys(trigger for candidate in candidates for trigger in candidate.repair_triggers)
    )
    if selected_candidate is not None:
        final_status = "passed"
        final_kernel_payload = _final_kernel_payload(
            config=cfg,
            transformed_adapter=transformed_adapter,
            transformed_adapter_signature=transformed_signature,
            base_adapter_signature=base_signature,
            identity_mass=identity_mass,
            identity_mass_signature=identity_mass_signature,
            selected_candidate=selected_candidate,
        )
    result_without_path = FixedTransportHMCKernelTuningResult(
        config=cfg,
        transformed_adapter_signature=transformed_signature,
        base_adapter_signature=base_signature,
        fixed_transport_manifest_hash=transformed_adapter.transport_manifest_hash,
        target_dimension=target_dimension,
        identity_z_mass_artifact_payload=identity_mass.to_payload(include_arrays=True),
        identity_z_mass_artifact_signature=identity_mass_signature,
        candidates=tuple(candidates),
        selected_candidate_index=selected_index,
        final_status=final_status,
        final_kernel_payload=final_kernel_payload,
        artifact_path=None,
        fixed_grid_scale_selection_payload=scale_payload,
        diagnostic_roles=_diagnostic_roles(),
        hard_vetoes=hard_vetoes,
        repair_triggers=repair_triggers,
    )
    artifact_path = _resolve_result_path(
        output_dir=output_dir,
        output_filename=cfg.output_filename,
    )
    if artifact_path is None:
        return result_without_path
    result_with_path = FixedTransportHMCKernelTuningResult(
        config=result_without_path.config,
        transformed_adapter_signature=result_without_path.transformed_adapter_signature,
        base_adapter_signature=result_without_path.base_adapter_signature,
        fixed_transport_manifest_hash=result_without_path.fixed_transport_manifest_hash,
        target_dimension=result_without_path.target_dimension,
        identity_z_mass_artifact_payload=result_without_path.identity_z_mass_artifact_payload,
        identity_z_mass_artifact_signature=result_without_path.identity_z_mass_artifact_signature,
        candidates=result_without_path.candidates,
        selected_candidate_index=result_without_path.selected_candidate_index,
        final_status=result_without_path.final_status,
        final_kernel_payload=result_without_path.final_kernel_payload,
        artifact_path=str(artifact_path),
        fixed_grid_scale_selection_payload=(
            result_without_path.fixed_grid_scale_selection_payload
        ),
        diagnostic_roles=result_without_path.diagnostic_roles,
        hard_vetoes=result_without_path.hard_vetoes,
        repair_triggers=result_without_path.repair_triggers,
        nonclaims=result_without_path.nonclaims,
    )
    _write_result_payload(path=artifact_path, result=result_with_path)
    return result_with_path


def _run_fixed_transport_candidate_attempts(
    config: FixedTransportHMCKernelTuningConfig,
    *,
    transformed_adapter: FixedTransportValueScoreAdapter,
    identity_mass: PrecomputedMassArtifact,
    initial_position: np.ndarray,
    target_dimension: int,
    run_full_chain: RunFullChainFn,
) -> tuple[list[FixedTransportHMCCandidateResult], Mapping[str, Any] | None]:
    base_steps = tuple(config.fixed_grid_base_step_size_candidates)
    scale_candidates = tuple(config.fixed_grid_scale_candidates)
    if not base_steps:
        candidates = _run_fixed_transport_candidate_grid(
            config,
            transformed_adapter=transformed_adapter,
            identity_mass=identity_mass,
            initial_position=initial_position,
            target_dimension=target_dimension,
            initial_step_size=config.initial_step_size,
            index_offset=0,
            seed_offset=0,
            run_full_chain=run_full_chain,
        )
        return candidates, None

    selection_payload: Mapping[str, Any] | None = None
    probe_acceptances: list[float | None] = []
    attempts: list[Mapping[str, Any]] = []
    pilot_leapfrog = (
        config.fixed_grid_num_leapfrog_steps
        if config.fixed_grid_num_leapfrog_steps is not None
        else max(config.leapfrog_grid)
    )
    max_attempts = min(len(scale_candidates), int(config.fixed_grid_max_attempts))
    selected_scale: float | None = None
    for attempt_index, scale in enumerate(scale_candidates[:max_attempts]):
        attempt_step = float(scale) * max(base_steps)
        probe = _run_fixed_grid_scale_probe(
            config,
            adapter=transformed_adapter,
            initial_position=initial_position,
            step_size=attempt_step,
            num_leapfrog_steps=int(pilot_leapfrog),
            attempt_index=attempt_index,
            run_full_chain=run_full_chain,
        )
        acceptance = _scale_probe_acceptance(probe)
        probe_acceptances.append(acceptance)
        selection = select_hmc_fixed_grid_scale(
            base_step_size_candidates=base_steps,
            num_leapfrog_step_candidates=config.leapfrog_grid,
            scale_candidates=scale_candidates[: len(probe_acceptances)],
            pilot_acceptance_rates=tuple(probe_acceptances),
            acceptance_band=config.acceptance_band,
            fallback_acceptance_max=config.fixed_grid_fallback_acceptance_max,
            target_acceptance=config.target_accept_prob,
            pilot_base_step_size=max(base_steps),
            pilot_num_leapfrog_steps=int(pilot_leapfrog),
            policy_source=f"{config.source}.fixed_grid_scale_repair",
        )
        attempts.append(
            {
                "attempt_index": int(attempt_index),
                "scale": float(scale),
                "initial_step_size": float(attempt_step),
                "pilot_num_leapfrog_steps": int(pilot_leapfrog),
                "pilot_acceptance_rate": acceptance,
                "acceptance_class": classify_hmc_fixed_grid_acceptance(
                    acceptance,
                    acceptance_band=config.acceptance_band,
                    fallback_acceptance_max=config.fixed_grid_fallback_acceptance_max,
                ),
                "probe_final_status": probe["final_status"],
                "probe_diagnostic_role": probe["diagnostic_role"],
                "probe_hard_vetoes": probe["hard_vetoes"],
                "probe_repair_triggers": probe["repair_triggers"],
                "probe_diagnostics": probe["diagnostics"],
                "selection_payload": selection.payload(),
            }
        )
        selection_payload = {
            "artifact_type": "bayesfilter_fixed_transport_hmc_grid_scale_repair",
            "schema_version": 1,
            "attempts": tuple(attempts),
            "latest_selection": selection.payload(),
            "selected_scale": selection.selected_scale,
            "status": "repair_attempts_exhausted",
            "nonclaims": FIXED_TRANSPORT_HMC_TUNING_NONCLAIMS,
        }
        if attempts[-1]["acceptance_class"] == "in_band":
            selected_scale = float(scale)
            selection_payload = {
                **selection_payload,
                "selected_scale": selected_scale,
                "status": "accepted_in_band",
            }
            break
    if selected_scale is None:
        return [], selection_payload
    selected_attempt = attempts[-1]
    verification = _run_verification(
        adapter=transformed_adapter,
        initial_position=initial_position,
        step_size=float(selected_attempt["initial_step_size"]),
        num_leapfrog_steps=int(pilot_leapfrog),
        candidate_index=0,
        config=config,
        run_full_chain=run_full_chain,
    )
    candidate = FixedTransportHMCCandidateResult(
        candidate_index=0,
        num_leapfrog_steps=int(pilot_leapfrog),
        ladder_result=None,
        verification_config_payload=verification["config_payload"],
        verification_diagnostics=verification["diagnostics"],
        final_status=verification["final_status"],
        diagnostic_role=verification["diagnostic_role"],
        fixed_kernel_step_size=float(selected_attempt["initial_step_size"]),
        hard_vetoes=verification["hard_vetoes"],
        repair_triggers=verification["repair_triggers"],
    )
    return [candidate], selection_payload


def _run_fixed_grid_scale_probe(
    config: FixedTransportHMCKernelTuningConfig,
    *,
    adapter: FixedTransportValueScoreAdapter,
    initial_position: np.ndarray,
    step_size: float,
    num_leapfrog_steps: int,
    attempt_index: int,
    run_full_chain: RunFullChainFn,
) -> Mapping[str, Any]:
    return _run_verification(
        adapter=adapter,
        initial_position=initial_position,
        step_size=float(step_size),
        num_leapfrog_steps=int(num_leapfrog_steps),
        candidate_index=10_000 + int(attempt_index),
        config=config,
        run_full_chain=run_full_chain,
        probe_only=True,
    )


def _scale_probe_acceptance(probe: Mapping[str, Any]) -> float | None:
    hard_vetoes = tuple(str(item) for item in probe.get("hard_vetoes", ()))
    non_acceptance_vetoes = tuple(
        item for item in hard_vetoes
        if item != "verification_acceptance_outside_repair_band"
    )
    if non_acceptance_vetoes:
        return None
    return _scalar_or_none(probe["diagnostics"].get("acceptance_rate"))


def _run_fixed_transport_candidate_grid(
    config: FixedTransportHMCKernelTuningConfig,
    *,
    transformed_adapter: FixedTransportValueScoreAdapter,
    identity_mass: PrecomputedMassArtifact,
    initial_position: np.ndarray,
    target_dimension: int,
    initial_step_size: float,
    index_offset: int,
    seed_offset: int,
    run_full_chain: RunFullChainFn,
) -> list[FixedTransportHMCCandidateResult]:
    candidates: list[FixedTransportHMCCandidateResult] = []
    for local_index, leapfrog in enumerate(config.leapfrog_grid):
        candidate_index = int(index_offset) + int(local_index)
        ladder_config = FixedMassHMCTuningBudgetLadderConfig(
            budget_schedule=config.budget_schedule,
            initial_step_size=float(initial_step_size),
            num_leapfrog_steps=int(leapfrog),
            target_accept_prob=config.target_accept_prob,
            acceptance_band=config.acceptance_band,
            repair_band=config.repair_band,
            step_repair_factor=config.step_repair_factor,
            step_repair_min_directional_factor=config.step_repair_min_directional_factor,
            tune_num_results=config.tune_num_results,
            screen_num_results=config.screen_num_results,
            screen_num_burnin_steps=config.screen_num_burnin_steps,
            tune_seed_base=_offset_seed(
                config.tune_seed_base,
                int(seed_offset) + int(local_index),
            ),
            screen_seed_base=_offset_seed(
                config.screen_seed_base,
                int(seed_offset) + int(local_index),
            ),
            chain_execution_mode=config.chain_execution_mode,
            use_xla=config.use_xla,
            target_scope=transformed_adapter.target_scope,
            target_status_trace_policy=config.target_status_trace_policy,
            source=config.source,
        )
        ladder = run_fixed_mass_hmc_tuning_budget_ladder(
            adapter=transformed_adapter,
            mass_artifact=identity_mass,
            initial_state_factory=_initial_state_factory(
                target_dimension=target_dimension,
                chain_count=config.chain_count,
            ),
            config=ladder_config,
            run_full_chain=run_full_chain,
        )
        if not ladder.passed:
            candidates.append(
                FixedTransportHMCCandidateResult(
                    candidate_index=candidate_index,
                    num_leapfrog_steps=int(leapfrog),
                    ladder_result=ladder,
                    verification_config_payload=None,
                    verification_diagnostics={},
                    final_status=f"ladder_{ladder.final_status}",
                    diagnostic_role="ladder_nonpass",
                    hard_vetoes=_collect_ladder_hard_vetoes(ladder),
                    repair_triggers=_collect_ladder_repair_triggers(ladder),
                )
            )
            continue
        selected_step = _selected_ladder_step(ladder)
        verification = _run_verification(
            adapter=transformed_adapter,
            initial_position=initial_position,
            step_size=selected_step,
            num_leapfrog_steps=int(leapfrog),
            candidate_index=candidate_index,
            config=config,
            run_full_chain=run_full_chain,
        )
        candidates.append(
            FixedTransportHMCCandidateResult(
                candidate_index=candidate_index,
                num_leapfrog_steps=int(leapfrog),
                ladder_result=ladder,
                verification_config_payload=verification["config_payload"],
                verification_diagnostics=verification["diagnostics"],
                final_status=verification["final_status"],
                diagnostic_role=verification["diagnostic_role"],
                hard_vetoes=verification["hard_vetoes"],
                repair_triggers=verification["repair_triggers"],
            )
        )
    return candidates


def _validate_base_adapter_authority(base_adapter: Any) -> None:
    capability = value_score_capability(base_adapter)
    if capability.value_score_authority in _FORBIDDEN_BASE_AUTHORITIES:
        raise ValueError(
            "fixed-transport HMC tuning requires reviewed analytical/custom "
            "base value/score authority; gradient_tape_fallback is forbidden"
        )


def _resolve_target_scope(
    base_adapter: Any,
    config: FixedTransportHMCKernelTuningConfig,
) -> str:
    capability = value_score_capability(base_adapter)
    if config.target_scope is not None:
        return str(config.target_scope)
    if capability.target_scope is None or not str(capability.target_scope):
        raise ValueError("target_scope is required when base adapter has none")
    return f"{capability.target_scope}:fixed_transport"


def _validate_initial_position(value: Any, parameter_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (int(parameter_dim),):
        raise ValueError("initial_position must have shape [parameter_dim]")
    if not np.all(np.isfinite(array)):
        raise ValueError("initial_position must be finite")
    return array


def _identity_z_mass_artifact(
    *,
    position: np.ndarray,
    adapter_signature: str,
) -> PrecomputedMassArtifact:
    dimension = int(position.shape[0])
    identity = np.eye(dimension, dtype=float)
    return PrecomputedMassArtifact(
        position=position,
        covariance=identity,
        factor=identity,
        adapter_signature=adapter_signature,
        position_role="fixed_neutra_initial_z",
        covariance_source="fixed_identity_z",
        matrix_used_for_square_root="identity_z",
        source="bayesfilter.fixed_transport_hmc_tuning.identity_z_mass",
        regularization_report={"regularization_applied": False},
        nonclaims=(
            "fixed identity mass in trained-transport z coordinates",
            "no residual mass adaptation claim",
            "no MAP quality claim",
            "no posterior convergence claim",
        ),
    )


def _initial_state_factory(
    *,
    target_dimension: int,
    chain_count: int,
) -> Callable[[tuple[int, int], str, int, int, float], np.ndarray]:
    def initial_state(
        _seed: tuple[int, int],
        _stage: str,
        _round_index: int,
        _budget: int,
        _step: float,
    ) -> np.ndarray:
        return np.zeros((int(chain_count), int(target_dimension)), dtype=float)

    return initial_state


def _run_verification(
    *,
    adapter: FixedTransportValueScoreAdapter,
    initial_position: np.ndarray,
    step_size: float,
    num_leapfrog_steps: int,
    candidate_index: int,
    config: FixedTransportHMCKernelTuningConfig,
    run_full_chain: RunFullChainFn,
    probe_only: bool = False,
) -> Mapping[str, Any]:
    seed_base = config.screen_seed_base if probe_only else config.verification_seed_base
    num_results = (
        config.screen_num_results if probe_only else config.verification_num_results
    )
    num_burnin_steps = (
        config.screen_num_burnin_steps
        if probe_only
        else config.verification_num_burnin_steps
    )
    verification_config = FullChainHMCConfig(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
        seed=_offset_seed(seed_base, candidate_index),
        use_xla=config.use_xla,
        trace_policy="standard",
        target_status_trace_policy=config.target_status_trace_policy,
        target_scope=adapter.target_scope,
        chain_execution_mode=config.chain_execution_mode,
    )
    initial_state = np.broadcast_to(
        np.asarray(initial_position, dtype=float),
        (config.chain_count, adapter.parameter_dim),
    ).copy()
    run_error: Exception | None = None
    try:
        run_result = run_full_chain(adapter, initial_state, verification_config)
        diagnostics = _verification_diagnostics(
            run_result,
            adapter=adapter,
            config=config,
            require_modern=bool(
                config.require_modern_rank_normalized_verification and not probe_only
            ),
        )
    except Exception as exc:  # noqa: BLE001 - verification is fail-closed.
        run_error = exc
        diagnostics = _error_diagnostics(exc)
    diagnostics = dict(diagnostics)
    diagnostics["diagnostic_context"] = (
        "fixed_transport_scale_selection_probe"
        if probe_only
        else "fixed_transport_fresh_verification"
    )
    diagnostics["probe_only"] = bool(probe_only)
    diagnostics["initial_state_shape"] = tuple(int(item) for item in initial_state.shape)
    diagnostics["rank2_chain_batched_initial_state"] = True
    diagnostics["nonclaims"] = FIXED_TRANSPORT_HMC_TUNING_NONCLAIMS
    final_status, diagnostic_role, hard_vetoes, repair_triggers = _classify_verification(
        config,
        diagnostics=diagnostics,
        run_error=run_error,
        require_modern=bool(
            config.require_modern_rank_normalized_verification and not probe_only
        ),
    )
    return {
        "config_payload": verification_config.signature_payload(),
        "diagnostics": diagnostics,
        "final_status": final_status,
        "diagnostic_role": diagnostic_role,
        "hard_vetoes": hard_vetoes,
        "repair_triggers": repair_triggers,
    }


def _verification_diagnostics(
    run_result: FullChainHMCRunResult,
    *,
    adapter: FixedTransportValueScoreAdapter,
    config: FixedTransportHMCKernelTuningConfig,
    require_modern: bool,
) -> Mapping[str, Any]:
    diagnostics = dict(run_result.diagnostics)
    trace = dict(run_result.trace)
    payload: dict[str, Any] = {
        "acceptance_rate": _scalar_or_none(diagnostics.get("acceptance_rate")),
        "binary_acceptance_rate": _scalar_or_none(
            diagnostics.get("binary_acceptance_rate")
        ),
        "finite_sample_count": _int_or_none(diagnostics.get("finite_sample_count")),
        "nonfinite_sample_count": _int_or_none(diagnostics.get("nonfinite_sample_count")),
        "final_step_size": _scalar_or_none(diagnostics.get("final_step_size")),
        "final_step_size_finite": _bool_or_none(diagnostics.get("final_step_size_finite")),
        "target_accept_prob": _scalar_or_none(diagnostics.get("target_accept_prob")),
        "trace_policy": diagnostics.get("trace_policy"),
        "divergence_status": diagnostics.get("divergence_status"),
        "divergence_count": _int_or_none(diagnostics.get("divergence_count")),
        "target_status_telemetry": _json_ready(
            diagnostics.get("target_status_telemetry")
        ),
        "runtime_metadata": _json_ready(run_result.metadata),
        "raw_diagnostics": _json_ready(diagnostics),
    }
    if "log_accept_ratio" in trace:
        log_accept = np.asarray(_tensor_to_numpy(trace["log_accept_ratio"]), dtype=float)
        finite = np.isfinite(log_accept)
        if np.any(finite):
            payload["acceptance_rate"] = float(
                np.mean(np.exp(np.minimum(log_accept[finite], 0.0)))
            )
            if "is_accepted" in trace:
                payload["binary_acceptance_rate"] = float(
                    np.mean(
                        np.asarray(
                            _tensor_to_numpy(trace["is_accepted"]),
                            dtype=bool,
                        )
                    )
                )
        payload["log_accept_ratio_finite"] = bool(np.all(finite))
        payload["max_abs_log_accept_ratio"] = (
            None if not np.any(finite) else float(np.max(np.abs(log_accept[finite])))
        )
    else:
        payload["log_accept_ratio_finite"] = None
    if "target_log_prob" in trace:
        target = np.asarray(_tensor_to_numpy(trace["target_log_prob"]), dtype=float)
        payload["target_log_prob_finite"] = bool(np.all(np.isfinite(target)))
    else:
        payload["target_log_prob_finite"] = None
    samples = np.asarray(_tensor_to_numpy(run_result.samples), dtype=float)
    finite_by_sample = np.all(np.isfinite(samples), axis=-1)
    payload["samples_all_finite"] = bool(np.all(finite_by_sample))
    payload["sample_shape"] = tuple(int(item) for item in samples.shape)
    if require_modern:
        raw_samples = _map_transformed_samples_to_target_coordinates(
            adapter,
            samples,
        )
        modern = rank_normalized_split_rhat_summary(
            raw_samples,
            rhat_max=config.verification_rhat_max,
        )
        payload["modern_rank_normalized_verification"] = _json_ready(modern)
        payload["modern_verification_coordinate_system"] = "raw_target_coordinates"
        payload["modern_verification_parameter_names"] = _parameter_names(
            adapter.base_adapter,
            adapter.parameter_dim,
        )
    else:
        payload["modern_rank_normalized_verification"] = None
        payload["modern_verification_coordinate_system"] = None
        payload["modern_verification_parameter_names"] = None
    return payload


def _classify_verification(
    config: FixedTransportHMCKernelTuningConfig,
    *,
    diagnostics: Mapping[str, Any],
    run_error: Exception | None,
    require_modern: bool,
) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    hard_vetoes: list[str] = []
    if run_error is not None:
        hard_vetoes.append("verification_hmc_error")
    acceptance = diagnostics.get("acceptance_rate")
    if acceptance is None or not _finite_number(acceptance):
        hard_vetoes.append("verification_acceptance_missing_or_nonfinite")
    if diagnostics.get("log_accept_ratio_finite") is not True:
        hard_vetoes.append("verification_log_accept_nonfinite_or_missing")
    maximum_energy = diagnostics.get("max_abs_log_accept_ratio")
    if not _finite_number(maximum_energy):
        hard_vetoes.append("verification_energy_error_missing_or_nonfinite")
    elif float(maximum_energy) > config.maximum_absolute_energy_error:
        hard_vetoes.append("verification_energy_error_exceeds_limit")
    if diagnostics.get("samples_all_finite") is not True:
        hard_vetoes.append("verification_samples_nonfinite_or_missing")
    if diagnostics.get("target_log_prob_finite") is False:
        hard_vetoes.append("verification_target_log_prob_nonfinite")
    divergence_count = _int_or_none(diagnostics.get("divergence_count"))
    if divergence_count is not None and divergence_count > 0:
        hard_vetoes.append("verification_native_divergence_detected")
    telemetry = diagnostics.get("target_status_telemetry")
    if config.target_status_trace_policy != "none":
        if not isinstance(telemetry, Mapping):
            hard_vetoes.append("verification_target_status_telemetry_missing")
        elif _bool_or_none(telemetry.get("telemetry_failure_veto")) is not False:
            hard_vetoes.append("verification_target_status_telemetry_failure")
    if require_modern:
        shape = diagnostics.get("sample_shape")
        if not isinstance(shape, (tuple, list)) or len(shape) != 3:
            hard_vetoes.append("verification_sample_shape_invalid_for_modern_rhat")
        else:
            if int(shape[0]) < config.verification_min_retained_results_per_chain:
                hard_vetoes.append("verification_retained_draw_count_below_minimum")
            if int(shape[1]) != 4:
                hard_vetoes.append("verification_chain_count_not_exactly_four")
        modern = diagnostics.get("modern_rank_normalized_verification")
        if not isinstance(modern, Mapping):
            hard_vetoes.append("verification_modern_rhat_missing")
        elif modern.get("passed") is not True:
            hard_vetoes.append("verification_modern_rank_folded_rhat_failed")
    if _finite_number(acceptance) and not (
        config.repair_band[0] <= float(acceptance) <= config.repair_band[1]
    ):
        hard_vetoes.append("verification_acceptance_outside_repair_band")
    if hard_vetoes:
        return "hard_veto", "hard_veto", tuple(dict.fromkeys(hard_vetoes)), ()
    acceptance_value = float(acceptance)
    if config.acceptance_band[0] <= acceptance_value <= config.acceptance_band[1]:
        return "passed", "fresh_fixed_kernel_verification_passed", (), ()
    return (
        "repair_or_retry",
        "verification_acceptance_repair_trigger",
        (),
        ("verification_acceptance_outside_pass_band_inside_repair_band",),
    )


def _select_candidate(
    candidates: Sequence[FixedTransportHMCCandidateResult],
    config: FixedTransportHMCKernelTuningConfig,
) -> int | None:
    passed = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if candidate.passed and candidate.selected_step_size is not None
    ]
    if not passed:
        return None
    target = float(config.target_accept_prob)
    selected_index, _candidate = min(
        passed,
        key=lambda item: (
            abs(float(item[1].selected_acceptance_rate) - target)
            if item[1].selected_acceptance_rate is not None
            else float("inf"),
            int(item[1].num_leapfrog_steps),
            int(item[1].candidate_index),
        ),
    )
    return int(selected_index)


def _final_kernel_payload(
    *,
    config: FixedTransportHMCKernelTuningConfig,
    transformed_adapter: FixedTransportValueScoreAdapter,
    transformed_adapter_signature: str,
    base_adapter_signature: str,
    identity_mass: PrecomputedMassArtifact,
    identity_mass_signature: str,
    selected_candidate: FixedTransportHMCCandidateResult,
) -> Mapping[str, Any]:
    step = selected_candidate.selected_step_size
    if step is None:
        raise ValueError("selected candidate must have a selected step size")
    return {
        "runtime": "bayesfilter.inference.tune_fixed_transport_hmc_kernel",
        "schema": "bayesfilter.fixed_transport_hmc_frozen_kernel_handoff.v1",
        "transformed_target_scope": transformed_adapter.target_scope,
        "base_adapter_signature": base_adapter_signature,
        "fixed_transport_manifest_hash": transformed_adapter.transport_manifest_hash,
        "transformed_adapter_signature": transformed_adapter_signature,
        "target_dimension": transformed_adapter.parameter_dim,
        "mass_policy": "fixed_identity_z",
        "identity_z_mass_artifact_payload": identity_mass.to_payload(include_arrays=True),
        "identity_z_mass_artifact_signature": identity_mass_signature,
        "step_size": float(step),
        "num_leapfrog_steps": int(selected_candidate.num_leapfrog_steps),
        "target_accept_prob": config.target_accept_prob,
        "acceptance_band": config.acceptance_band,
        "maximum_absolute_energy_error": config.maximum_absolute_energy_error,
        "proposal_dynamics_identity": config.proposal_dynamics_identity,
        "selected_candidate_index": selected_candidate.candidate_index,
        "selected_candidate_artifact_hash": selected_candidate.artifact_hash,
        "verification_config_payload": selected_candidate.verification_config_payload,
        "verification_diagnostics": selected_candidate.verification_diagnostics,
        "modern_rank_normalized_verification_required": (
            config.require_modern_rank_normalized_verification
        ),
        "modern_rhat_definition": (
            RANK_NORMALIZED_SPLIT_RHAT_DEFINITION
            if config.require_modern_rank_normalized_verification
            else None
        ),
        "rank2_chain_batched_target_required": True,
        "windowed_mass_adaptation_used": False,
        "mass_adaptation_used": False,
        "transport_training_or_adaptation_used": False,
        "nonclaims": FIXED_TRANSPORT_HMC_TUNING_NONCLAIMS,
    }


def _resolve_result_path(
    *,
    output_dir: str | Path | None,
    output_filename: str,
) -> Path | None:
    if output_dir is None:
        return None
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / output_filename


def _write_result_payload(
    *,
    path: Path,
    result: FixedTransportHMCKernelTuningResult,
) -> None:
    payload = _json_ready(result.payload())
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_adapter_signature(adapter: Any) -> str:
    explicit = getattr(adapter, "adapter_signature", None)
    if explicit is not None:
        return str(explicit() if callable(explicit) else explicit)
    return stable_adapter_signature(adapter)


def _mass_artifact_signature(mass_artifact: PrecomputedMassArtifact) -> str:
    return stable_config_hash(
        {
            "signature_payload": mass_artifact.signature_payload(),
            "position": np.asarray(mass_artifact.position, dtype=float),
            "covariance": np.asarray(mass_artifact.covariance, dtype=float),
            "factor": np.asarray(mass_artifact.factor, dtype=float),
        }
    )


def _selected_ladder_step(ladder: FixedMassHMCTuningBudgetLadderResult) -> float:
    selected = ladder.selected_round
    if selected is None or selected.tuned_step_size is None:
        raise ValueError("passed ladder must have a selected tuned step")
    value = float(selected.tuned_step_size)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("selected tuned step must be positive and finite")
    return value


def _collect_ladder_hard_vetoes(
    ladder: FixedMassHMCTuningBudgetLadderResult,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(veto for round_result in ladder.rounds for veto in round_result.hard_vetoes)
    )


def _collect_ladder_repair_triggers(
    ladder: FixedMassHMCTuningBudgetLadderResult,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            trigger for round_result in ladder.rounds for trigger in round_result.repair_triggers
        )
    )


def _diagnostic_roles() -> Mapping[str, str]:
    return {
        "base_value_score_authority": "hard_veto_if_gradient_tape_fallback",
        "fixed_transport_manifest_hash": "artifact_identity",
        "identity_z_mass_policy": "hard_boundary",
        "candidate_ladder": "step_tuning_screen",
        "fresh_verification": "handoff_promotion_screen",
        "acceptance": "promotion_screen_and_repair_trigger",
        "modern_rank_folded_rhat": "required_when_configured_handoff_promotion_screen",
        "runtime": "explanatory_diagnostic",
    }


def _map_transformed_samples_to_target_coordinates(
    adapter: FixedTransportValueScoreAdapter,
    samples: np.ndarray,
) -> np.ndarray:
    if samples.ndim != 3:
        raise ValueError("verification samples must have shape [draw, chain, parameter]")
    draw_count, chain_count, parameter_dim = samples.shape
    if parameter_dim != adapter.parameter_dim:
        raise ValueError("verification sample parameter dimension mismatch")
    flat = tf.convert_to_tensor(
        samples.reshape(draw_count * chain_count, parameter_dim),
        dtype=tf.float64,
    )
    raw = adapter.transport.forward_batch(flat)
    raw_array = np.asarray(_tensor_to_numpy(raw), dtype=float)
    if raw_array.shape != (draw_count * chain_count, parameter_dim):
        raise ValueError("fixed transport raw verification shape mismatch")
    return raw_array.reshape(draw_count, chain_count, parameter_dim)


def _parameter_names(adapter: Any, parameter_dim: int) -> tuple[str, ...]:
    names = getattr(adapter, "parameter_names", None)
    if names is not None:
        values = names() if callable(names) else names
        result = tuple(str(item) for item in values)
        if len(result) == parameter_dim:
            return result
    return tuple(f"parameter_{index}" for index in range(parameter_dim))


def _offset_seed(seed: tuple[int, int], offset: int) -> tuple[int, int]:
    return int(seed[0]), int(seed[1]) + int(offset)


def _validate_seed(seed: Sequence[int]) -> tuple[int, int]:
    values = tuple(int(item) for item in seed)
    if len(values) != 2:
        raise ValueError("seed must contain exactly two integers")
    return values


def _validate_band(values: Sequence[float], *, name: str) -> tuple[float, float]:
    raw = tuple(values)
    if len(raw) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower, upper = tuple(float(item) for item in raw)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"{name} values must be finite")
    if not 0.0 < lower <= upper < 1.0:
        raise ValueError(f"{name} must satisfy 0 < lower <= upper < 1")
    return lower, upper


def _validate_step_repair_multiplier(value: Any, *, name: str) -> float:
    multiplier = float(value)
    if not np.isfinite(multiplier) or multiplier <= 1.0:
        raise ValueError(f"{name} must be finite and greater than 1")
    return multiplier


def _string_tuple(values: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(values, str):
        raw = (values,)
    else:
        raw = tuple(values)
    return tuple(str(item) for item in raw if str(item))


def _finite_number(value: Any) -> bool:
    scalar = _scalar_or_none(value)
    return scalar is not None and bool(np.isfinite(scalar))


def _tensor_to_numpy(value: Any) -> Any:
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _scalar_or_none(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(_tensor_to_numpy(value))
    if array.size == 0:
        return None
    try:
        return float(array.reshape(-1)[-1])
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    scalar = _scalar_or_none(value)
    return None if scalar is None else int(scalar)


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    array = np.asarray(_tensor_to_numpy(value))
    if array.size == 0:
        return None
    return bool(array.reshape(-1)[-1])


def _json_ready(value: Any) -> Any:
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value
