"""TensorFlow/TFP tuning for HMC behind a frozen nonlinear transport.

The public tuner owns the trajectory grid and tunes one scalar step size over
the complete rank-2 chain bank. It intentionally has no NumPy, NUTS, or mass
adaptation dependency. Tuning transitions are discarded and every candidate is
checked with a fresh fixed kernel before it can be handed to sequential HMC.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
    FixedTransportFullChainConfig as _FullChainHMCConfig,
    FixedTransportHMCPolicy as _TuningPolicy,
    RunFullChainFn,
    build_fixed_transport_value_score_adapter,
    fixed_transport_target_status_diagnostics as _target_status_diagnostics,
    fixed_transport_tensor_diagnostics as _tensor_diagnostics,
    run_fixed_transport_full_chain_tfp_hmc as _run_full_chain_tfp_hmc,
)
from bayesfilter.inference.hmc_convergence import (
    RANK_NORMALIZED_SPLIT_RHAT_DEFINITION,
    rank_normalized_split_rhat_summary,
)
from bayesfilter.inference.posterior_adapter import value_score_capability
from bayesfilter.inference.tuning_contract import require_active_hmc_tuning_route


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
    verification_coordinate_system: str = "raw_target_coordinates"
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
        for name in ("step_repair_factor", "step_repair_min_directional_factor"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 1.0:
                raise ValueError(f"{name} must be finite and greater than 1")
            object.__setattr__(self, name, value)
        budgets = tuple(int(value) for value in self.budget_schedule)
        if not budgets or any(value <= 0 for value in budgets):
            raise ValueError("budget_schedule must contain positive integers")
        object.__setattr__(self, "budget_schedule", budgets)
        for name in ("tune_seed_base", "screen_seed_base", "verification_seed_base"):
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
        if bool(self.require_modern_rank_normalized_verification):
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
        if not math.isfinite(fallback) or not acceptance[1] <= fallback < 1.0:
            raise ValueError("fixed_grid_fallback_acceptance_max is invalid")
        object.__setattr__(self, "fixed_grid_fallback_acceptance_max", fallback)

    def payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["maximum_absolute_energy_error_role"] = "explanatory_alert_only"
        payload["shared_scalar_step_across_chain_bank"] = True
        payload["runtime_numerical_backend"] = "tensorflow_tfp_only"
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
            "schema": "bayesfilter.fixed_transport_hmc_kernel_tuning_result.v2",
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
            "diagnostic_roles": self.diagnostic_roles or {},
            "hard_vetoes": self.hard_vetoes,
            "repair_triggers": self.repair_triggers,
            "passed": self.passed,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "reports_default_readiness": False,
            "nonclaims": self.nonclaims,
        }


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

    require_active_hmc_tuning_route("tune_fixed_transport_hmc_kernel")
    cfg = config or FixedTransportHMCKernelTuningConfig(initial_step_size=0.1)
    if not isinstance(cfg, FixedTransportHMCKernelTuningConfig):
        raise TypeError("config must be FixedTransportHMCKernelTuningConfig")
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
    mass_payload = _identity_mass_payload(z0, transformed_signature)
    mass_signature = _stable_hash(mass_payload)
    candidates, scale_payload = _candidate_attempts(
        cfg,
        adapter=adapter,
        z0=z0,
        run_full_chain=run_full_chain,
        passthrough_exceptions=passthrough,
    )
    selected = _select_candidate(candidates, cfg)
    selected_candidate = None if selected is None else candidates[selected]
    final_kernel = None
    if selected_candidate is not None:
        final_kernel = {
            "schema": "bayesfilter.fixed_transport_hmc_kernel.v2",
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
            "rank2_chain_batched_target_required": True,
            "shared_scalar_step_across_chain_bank": True,
            "use_xla": cfg.use_xla,
            "proposal_dynamics_identity": cfg.proposal_dynamics_identity,
            "windowed_mass_adaptation_used": False,
            "mass_adaptation_used": False,
            "transport_training_or_adaptation_used": False,
            "verification_diagnostics": selected_candidate.verification_diagnostics,
            "nonclaims": FIXED_TRANSPORT_HMC_TUNING_NONCLAIMS,
        }
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
        final_status="passed" if selected is not None else "no_viable_candidate",
        final_kernel_payload=final_kernel,
        fixed_grid_scale_selection_payload=scale_payload,
        diagnostic_roles=_diagnostic_roles(),
        hard_vetoes=tuple(
            dict.fromkeys(veto for candidate in candidates for veto in candidate.hard_vetoes)
        ),
        repair_triggers=tuple(
            dict.fromkeys(
                trigger for candidate in candidates for trigger in candidate.repair_triggers
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
    state = _initial_state(config, adapter.parameter_dim)
    step = config.initial_step_size
    rounds = []
    selected_step = None
    repair_triggers: list[str] = []
    hard_vetoes: list[str] = []
    for round_index, budget in enumerate(config.budget_schedule):
        tune_config = _chain_config(
            config,
            num_results=config.tune_num_results,
            burnin=budget,
            step=step,
            leapfrog=leapfrog,
            seed=_offset_seed(config.tune_seed_base, candidate_index * 100 + round_index),
            adaptive_steps=budget,
        )
        try:
            tune_result = run_full_chain(adapter, state, tune_config)
            tune_diagnostics = _verification_diagnostics(
                tune_result, adapter=adapter, config=config, require_modern=False
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
                "initial_step_size": step if round_index else config.initial_step_size,
                "tuned_step_size": tuned_step,
                "tune_config": tune_config.signature_payload(),
                "tune_diagnostics": tune_diagnostics,
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
) -> Mapping[str, Any]:
    chain_config = _chain_config(
        config,
        num_results=config.screen_num_results if probe_only else config.verification_num_results,
        burnin=(
            config.screen_num_burnin_steps
            if probe_only
            else config.verification_num_burnin_steps
        ),
        step=step,
        leapfrog=leapfrog,
        seed=_offset_seed(
            config.screen_seed_base if probe_only else config.verification_seed_base,
            candidate_index,
        ),
        adaptive_steps=0,
    )
    initial_state = _initial_state(config, adapter.parameter_dim)
    error = None
    try:
        result = run_full_chain(adapter, initial_state, chain_config)
        diagnostics = _verification_diagnostics(
            result,
            adapter=adapter,
            config=config,
            require_modern=(config.require_modern_rank_normalized_verification and not probe_only),
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
                "fixed_transport_scale_selection_probe"
                if probe_only
                else "fixed_transport_fresh_verification"
            ),
            "probe_only": probe_only,
            "initial_state_shape": tuple(int(value) for value in initial_state.shape),
            "initial_state_all_zero": bool(tf.reduce_all(initial_state == 0.0).numpy()),
            "initial_state_bank": _json_ready(initial_state),
            "rank2_chain_batched_initial_state": True,
            "nonclaims": FIXED_TRANSPORT_HMC_TUNING_NONCLAIMS,
        }
    )
    status, role, vetoes, repairs = _classify_verification(
        config,
        diagnostics=diagnostics,
        run_error=error,
        require_modern=(config.require_modern_rank_normalized_verification and not probe_only),
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
    require_modern: bool,
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
    payload["sample_shape"] = tuple(int(value) for value in samples.shape)
    payload["modern_rank_normalized_verification"] = None
    payload["modern_verification_coordinate_system"] = None
    if require_modern:
        raw = (
            samples
            if config.verification_coordinate_system == "hmc_coordinates"
            else _map_samples(adapter, samples)
        )
        modern = rank_normalized_split_rhat_summary(
            raw, rhat_max=config.verification_rhat_max
        )
        payload["modern_rank_normalized_verification"] = modern
        payload["modern_verification_coordinate_system"] = config.verification_coordinate_system
    return payload


def _classify_verification(
    config: FixedTransportHMCKernelTuningConfig,
    *,
    diagnostics: Mapping[str, Any],
    run_error: Exception | None,
    require_modern: bool,
) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    hard = []
    if run_error is not None:
        hard.append("verification_runtime_error")
    acceptance = _scalar_or_none(diagnostics.get("acceptance_rate"))
    if acceptance is None or not math.isfinite(acceptance):
        hard.append("verification_acceptance_missing_or_nonfinite")
    elif not config.acceptance_band[0] <= acceptance <= config.acceptance_band[1]:
        hard.append("verification_acceptance_outside_pass_band")
    hard.extend(_basic_hard_vetoes(diagnostics, prefix="verification"))
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
    hard = list(dict.fromkeys(hard))
    repairs = []
    if acceptance is not None and math.isfinite(acceptance):
        if acceptance < config.acceptance_band[0]:
            repairs.append("verification_acceptance_below_pass_band")
        elif acceptance > config.acceptance_band[1]:
            repairs.append("verification_acceptance_above_pass_band")
    return (
        "passed" if not hard else "hard_veto",
        "fresh_fixed_kernel_verification" if not hard else "hard_veto",
        tuple(hard),
        tuple(repairs),
    )


def _basic_hard_vetoes(diagnostics: Mapping[str, Any], *, prefix: str) -> list[str]:
    hard = []
    for key, reason in (
        ("samples_all_finite", "samples_nonfinite_or_missing"),
        ("log_accept_ratio_finite", "log_accept_nonfinite_or_missing"),
        ("target_log_prob_finite", "target_log_prob_nonfinite_or_missing"),
    ):
        if diagnostics.get(key) is not True:
            hard.append(f"{prefix}_{reason}")
    if diagnostics.get("proposed_target_log_prob_finite") is False:
        hard.append(f"{prefix}_proposed_target_log_prob_nonfinite")
    if diagnostics.get("target_score_finite") is False:
        hard.append(f"{prefix}_target_score_nonfinite")
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
    config: FixedTransportHMCKernelTuningConfig, parameter_dim: int
) -> tf.Tensor:
    if config.initial_state_bank:
        state = tf.convert_to_tensor(config.initial_state_bank, tf.float64)
        if state.shape != (config.chain_count, int(parameter_dim)):
            raise ValueError(
                "initial_state_bank must have shape [chain_count, parameter_dim]"
            )
        return state
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


def _acceptance_class(
    acceptance: float | None, config: FixedTransportHMCKernelTuningConfig
) -> str:
    if acceptance is None or not math.isfinite(acceptance):
        return "invalid"
    if config.acceptance_band[0] <= acceptance <= config.acceptance_band[1]:
        return "in_band"
    if acceptance < config.acceptance_band[0]:
        return "below_band"
    return "above_band"


def _diagnostic_roles() -> Mapping[str, str]:
    return {
        "base_value_score_authority": "hard_veto_if_gradient_tape_fallback",
        "fixed_transport_manifest_hash": "artifact_identity",
        "identity_z_mass_policy": "hard_boundary",
        "candidate_ladder": "step_tuning_screen",
        "fresh_verification": "handoff_promotion_screen",
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


def _offset_seed(seed: tuple[int, int], offset: int) -> tuple[int, int]:
    return seed[0], seed[1] + int(offset)


def _string_tuple(value: Sequence[str] | str) -> tuple[str, ...]:
    values = (value,) if isinstance(value, str) else tuple(value)
    return tuple(str(item) for item in values if str(item))


__all__ = [
    "FIXED_TRANSPORT_HMC_TUNING_NONCLAIMS",
    "FixedTransportHMCCandidateResult",
    "FixedTransportHMCKernelTuningConfig",
    "FixedTransportHMCKernelTuningResult",
    "tune_fixed_transport_hmc_kernel",
]
