"""Fixed-mass bootstrap screening and bounded epsilon repair.

This preparation stage issues no final tuning or posterior authority.
Historical hmc_kernel_tuning imports remain aliases to these definitions.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Mapping

from bayesfilter.inference.batched_value_score import (
    _call_mapping_with_batch_rank_bridge,
    _call_value_score_with_batch_rank_bridge,
)
from bayesfilter.inference.hmc import (
    FullChainHMCConfig, FullChainHMCRunResult, PrecomputedMassArtifact,
    build_reusable_full_chain_tfp_hmc_runner, run_full_chain_tfp_hmc,
    program_signature, stable_adapter_signature,
)
from bayesfilter.inference.hmc_diagnostics import screen_hmc_diagnostics
from bayesfilter.inference.hmc_geometry import (
    HMCGeometryInitializationResult, _GEOMETRY_MAX_LEAPFROG,
    _GEOMETRY_MIN_LEAPFROG, _derive_seed, _validate_max_leapfrog_steps,
)
from bayesfilter.inference.hmc_warmup import (
    G2PreboundarySeedUseRegistry, _G2SeedRegistryError,
    _G2_BOOTSTRAP_ROUND_SEED_INTERFACE_HOPS_CONTRACT,
    g2_preboundary_shared_invalidity_exception,
)
from bayesfilter.inference.posterior_adapter import (
    ValueScoreCapability, value_score_capability,
)
from bayesfilter.runtime import stable_config_hash
from bayesfilter.inference.hmc_preparation_common import (
    _validate_band,
    _validate_seed,
    _validate_step_repair_multiplier,
    _string_tuple,
    _mass_artifact_signature,
    _round_seed,
    _scalar_or_none,
    _seed_from_mapping,
    _bool_or_none,
    _int_or_none,
    _json_ready,
    _runtime_seconds_or_none,
    _telemetry_payload,
    _finite_number,
)

BOOTSTRAP_SCREEN_NONCLAIMS = (
    "bootstrap fixed-kernel HMC screen only",
    "acceptance is a bootstrap repair diagnostic only",
    "no mass adaptation validity claim",
    "no trajectory tuning claim",
    "no posterior convergence claim",
    "no sampler superiority claim",
    "no default-readiness claim",
    "no GPU or XLA readiness claim",
)

BOOTSTRAP_ACCEPTANCE_STATISTIC = "mean_metropolis_probability_over_recorded_proposals"


RunFullChainFn = Callable[[Any, Any, FullChainHMCConfig], FullChainHMCRunResult]


BootstrapProgressCallback = Callable[[str, Mapping[str, Any]], None]


PrivateTuningDiagnosticCallback = Callable[[str, Mapping[str, Any]], None]


_G2_BOOTSTRAP_ROUND_SEED_DERIVATION_SITE_ID = (
    "hmc_kernel_tuning.run_hmc_bootstrap_screen.round_seed_derivation.v1"
)


_G2_BOOTSTRAP_ROUND_SEED_GATE_SITE_ID = (
    "hmc_kernel_tuning.run_hmc_bootstrap_screen.round_seed_gate.v1"
)


_G2_BOOTSTRAP_ROUND_SEED_INTERFACE_HOPS = (
    _G2_BOOTSTRAP_ROUND_SEED_INTERFACE_HOPS_CONTRACT
)


@dataclass(frozen=True)
class HMCBootstrapScreenConfig:
    """Policy-level config for the Phase 3 bootstrap screen.

    Model clients do not supply HMC mechanics here.  Step size and leapfrog
    count are derived internally from the Phase 2 geometry artifact and the
    bounded bootstrap repair rule.
    """

    target_accept_prob: float = 0.70
    acceptance_band: tuple[float, float] = (0.65, 0.75)
    repair_band: tuple[float, float] = (0.55, 0.85)
    step_repair_factor: float = 2.0
    max_leapfrog_steps: int = _GEOMETRY_MAX_LEAPFROG
    max_repairs: int = 5
    screen_num_results: int = 16
    screen_num_burnin_steps: int = 4
    seed: tuple[int, int] = (20260621, 3)
    chain_execution_mode: str = "tf_function"
    use_xla: bool = False
    target_scope: str | None = None
    target_status_trace_policy: str = "none"
    source: str = "bayesfilter.inference.hmc_kernel_tuning.bootstrap_screen"
    acceptance_role: str = "fixed_kernel_screen"

    def __post_init__(self) -> None:
        if self.acceptance_role not in {"fixed_kernel_screen", "warmup_startup_only"}:
            raise ValueError("unknown bootstrap acceptance role")
        target = float(self.target_accept_prob)
        if not math.isfinite(target) or not 0.0 < target < 1.0:
            raise ValueError("target_accept_prob must be finite and in (0, 1)")
        object.__setattr__(self, "target_accept_prob", target)
        acceptance_band = _validate_band(self.acceptance_band, name="acceptance_band")
        repair_band = _validate_band(self.repair_band, name="repair_band")
        if repair_band[0] > acceptance_band[0] or repair_band[1] < acceptance_band[1]:
            raise ValueError("repair_band must contain acceptance_band")
        object.__setattr__(self, "acceptance_band", acceptance_band)
        object.__setattr__(self, "repair_band", repair_band)
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
            "max_leapfrog_steps",
            _validate_max_leapfrog_steps(self.max_leapfrog_steps),
        )
        max_repairs = int(self.max_repairs)
        if max_repairs < 0:
            raise ValueError("max_repairs must be non-negative")
        object.__setattr__(self, "max_repairs", max_repairs)
        for name in ("screen_num_results", "screen_num_burnin_steps"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "seed", _validate_seed(self.seed))
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
        target_status_policy = str(self.target_status_trace_policy)
        if target_status_policy not in {"none", "per_chain_step"}:
            raise ValueError(
                "target_status_trace_policy must be 'none' or 'per_chain_step'"
            )
        object.__setattr__(self, "target_status_trace_policy", target_status_policy)
        source = str(self.source)
        if not source:
            raise ValueError("source must be non-empty")
        object.__setattr__(self, "source", source)

    def payload(self) -> Mapping[str, Any]:
        return {
            "target_accept_prob": self.target_accept_prob,
            "acceptance_band": self.acceptance_band,
            "repair_band": self.repair_band,
            "step_repair_factor": self.step_repair_factor,
            "max_leapfrog_steps": self.max_leapfrog_steps,
            "max_repairs": self.max_repairs,
            "screen_num_results": self.screen_num_results,
            "screen_num_burnin_steps": self.screen_num_burnin_steps,
            "seed": self.seed,
            "chain_execution_mode": self.chain_execution_mode,
            "use_xla": self.use_xla,
            "target_scope": self.target_scope,
            "target_status_trace_policy": self.target_status_trace_policy,
            "source": self.source,
            "acceptance_role": self.acceptance_role,
            "acceptance_statistic": BOOTSTRAP_ACCEPTANCE_STATISTIC,
        }


@dataclass(frozen=True)
class HMCBootstrapRepairRound:
    """One Phase 3 fixed-kernel screen attempt and repair decision."""

    round_index: int
    seed: tuple[int, int]
    step_size: float
    num_leapfrog_steps: int
    unclamped_num_leapfrog_steps: int
    target_trajectory_length: float
    leapfrog_clamped: bool
    clamp_direction: str | None
    classification: str
    diagnostic_role: str
    screen_config_payload: Mapping[str, Any] | None
    diagnostics: Mapping[str, Any]
    repair_action: str | None = None
    hard_vetoes: tuple[str, ...] = ()
    repair_triggers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "round_index", int(self.round_index))
        object.__setattr__(self, "seed", _validate_seed(self.seed))
        step = float(self.step_size)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be positive and finite")
        object.__setattr__(self, "step_size", step)
        leapfrogs = int(self.num_leapfrog_steps)
        raw_leapfrogs = int(self.unclamped_num_leapfrog_steps)
        if leapfrogs <= 0 or raw_leapfrogs <= 0:
            raise ValueError("leapfrog counts must be positive")
        object.__setattr__(self, "num_leapfrog_steps", leapfrogs)
        object.__setattr__(self, "unclamped_num_leapfrog_steps", raw_leapfrogs)
        trajectory = float(self.target_trajectory_length)
        if not math.isfinite(trajectory) or trajectory <= 0.0:
            raise ValueError("target_trajectory_length must be positive and finite")
        object.__setattr__(self, "target_trajectory_length", trajectory)
        object.__setattr__(self, "leapfrog_clamped", bool(self.leapfrog_clamped))
        clamp = None if self.clamp_direction is None else str(self.clamp_direction)
        if clamp not in {None, "min", "max"}:
            raise ValueError("clamp_direction must be None, 'min', or 'max'")
        object.__setattr__(self, "clamp_direction", clamp)
        object.__setattr__(self, "classification", str(self.classification))
        object.__setattr__(self, "diagnostic_role", str(self.diagnostic_role))
        payload = (
            None
            if self.screen_config_payload is None
            else dict(self.screen_config_payload)
        )
        object.__setattr__(self, "screen_config_payload", payload)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        action = None if self.repair_action is None else str(self.repair_action)
        object.__setattr__(self, "repair_action", action)
        object.__setattr__(self, "hard_vetoes", _string_tuple(self.hard_vetoes))
        object.__setattr__(self, "repair_triggers", _string_tuple(self.repair_triggers))

    @property
    def passed(self) -> bool:
        return self.classification == "passed"

    def payload(self) -> Mapping[str, Any]:
        return {
            "round_index": self.round_index,
            "seed": self.seed,
            "step_size": self.step_size,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "unclamped_num_leapfrog_steps": self.unclamped_num_leapfrog_steps,
            "target_trajectory_length": self.target_trajectory_length,
            "leapfrog_clamped": self.leapfrog_clamped,
            "clamp_direction": self.clamp_direction,
            "classification": self.classification,
            "diagnostic_role": self.diagnostic_role,
            "screen_config_payload": self.screen_config_payload,
            "diagnostics": self.diagnostics,
            "repair_action": self.repair_action,
            "hard_vetoes": self.hard_vetoes,
            "repair_triggers": self.repair_triggers,
        }


@dataclass(frozen=True)
class HMCBootstrapScreenResult:
    """Phase 3 bootstrap artifact; not final HMC tuning evidence."""

    config: HMCBootstrapScreenConfig
    geometry_artifact_hash: str
    adapter_signature: str
    hmc_adapter_signature: str
    mass_artifact_signature: str
    target_dimension: int
    rounds: tuple[HMCBootstrapRepairRound, ...]
    selected_round_index: int | None
    final_status: str
    seed_report: Mapping[str, Any]
    diagnostic_roles: Mapping[str, str]
    bootstrap_runner_route: Mapping[str, Any] | None = None
    nonclaims: tuple[str, ...] = BOOTSTRAP_SCREEN_NONCLAIMS

    def __post_init__(self) -> None:
        for name in (
            "geometry_artifact_hash",
            "adapter_signature",
            "hmc_adapter_signature",
            "mass_artifact_signature",
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
        rounds = tuple(self.rounds)
        if not rounds:
            raise ValueError("bootstrap result requires at least one round")
        object.__setattr__(self, "rounds", rounds)
        selected = (
            None if self.selected_round_index is None else int(self.selected_round_index)
        )
        if selected is not None and selected not in range(len(rounds)):
            raise ValueError("selected_round_index must refer to a round")
        if selected is not None and not rounds[selected].passed:
            raise ValueError("selected round must have passed")
        object.__setattr__(self, "selected_round_index", selected)
        object.__setattr__(self, "seed_report", dict(self.seed_report))
        object.__setattr__(self, "diagnostic_roles", dict(self.diagnostic_roles))
        route = (
            {}
            if self.bootstrap_runner_route is None
            else dict(self.bootstrap_runner_route)
        )
        object.__setattr__(self, "bootstrap_runner_route", route)
        nonclaims = tuple(str(item) for item in self.nonclaims)
        if not nonclaims:
            raise ValueError("nonclaims must be non-empty")
        object.__setattr__(self, "nonclaims", nonclaims)

    @property
    def passed(self) -> bool:
        return self.selected_round_index is not None

    @property
    def selected_round(self) -> HMCBootstrapRepairRound | None:
        if self.selected_round_index is None:
            return None
        return self.rounds[self.selected_round_index]

    @property
    def selected_kernel_payload(self) -> Mapping[str, Any] | None:
        selected = self.selected_round
        if selected is None:
            return None
        return _bootstrap_selected_kernel_payload(
            config=self.config,
            selected=selected,
            adapter_signature=self.adapter_signature,
            hmc_adapter_signature=self.hmc_adapter_signature,
            mass_artifact_signature=self.mass_artifact_signature,
            geometry_artifact_hash=self.geometry_artifact_hash,
            nonclaims=self.nonclaims,
        )

    @property
    def selected_kernel_hash(self) -> str | None:
        payload = self.selected_kernel_payload
        return None if payload is None else stable_config_hash(payload)

    @property
    def artifact_hash(self) -> str:
        return stable_config_hash(self.payload())

    def payload(self) -> Mapping[str, Any]:
        selected_payload = self.selected_kernel_payload
        return {
            "schema": "bayesfilter.hmc_bootstrap_screen.v1",
            "config": self.config.payload(),
            "geometry_artifact_hash": self.geometry_artifact_hash,
            "adapter_signature": self.adapter_signature,
            "hmc_adapter_signature": self.hmc_adapter_signature,
            "mass_artifact_signature": self.mass_artifact_signature,
            "target_dimension": self.target_dimension,
            "rounds": tuple(round_result.payload() for round_result in self.rounds),
            "selected_round_index": self.selected_round_index,
            "selected_kernel_payload": selected_payload,
            "selected_kernel_hash": None
            if selected_payload is None
            else stable_config_hash(selected_payload),
            "final_status": self.final_status,
            "seed_report": self.seed_report,
            "diagnostic_roles": self.diagnostic_roles,
            "bootstrap_runner_route": self.bootstrap_runner_route,
            "passed": self.passed,
            "reports_mass_adaptation_validity": False,
            "reports_trajectory_tuning": False,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "reports_gpu_or_xla_readiness": False,
            "nonclaims": self.nonclaims,
            "artifact_hash_components": {
                "hash_function": "bayesfilter.runtime.stable_config_hash",
                "json_normalization": "sort_keys_compact_json",
            },
        }


class _BootstrapFixedMassLatentValueScoreAdapter:
    """Latent fixed-mass target with a mass-bound stable adapter signature."""

    def __init__(
        self,
        *,
        base_adapter: Any,
        transform: Any,
        target_scope: str,
        adapter_signature: str,
        nonclaims: Sequence[str] = BOOTSTRAP_SCREEN_NONCLAIMS,
    ) -> None:
        if not hasattr(base_adapter, "log_prob_and_grad"):
            raise TypeError("base_adapter must expose log_prob_and_grad")
        self.base_adapter = base_adapter
        self.supports_retained_draw_batch = bool(
            getattr(base_adapter, "supports_retained_draw_batch", False)
        )
        self.supports_retained_flat_batch = bool(
            getattr(base_adapter, "supports_retained_flat_batch", False)
        )
        self.supports_retained_value_score_status = bool(
            getattr(base_adapter, "supports_retained_value_score_status", False)
            and callable(getattr(base_adapter, "log_prob_and_grad_status", None))
        )
        self.transform = transform
        self.parameter_dim = int(transform.dimension)
        self.target_scope = str(target_scope)
        if not self.target_scope:
            raise ValueError("target_scope must be non-empty")
        self.runtime_backend = (
            "bayesfilter.inference.hmc_kernel_tuning."
            "_BootstrapFixedMassLatentValueScoreAdapter"
        )
        self.nonclaims = _string_tuple(nonclaims)
        self._adapter_signature = str(adapter_signature)
        if not self._adapter_signature:
            raise ValueError("bootstrap fixed-mass adapter signature must be non-empty")

    def adapter_signature(self) -> str:
        return self._adapter_signature

    def value_score_capability(self) -> ValueScoreCapability:
        base_capability = value_score_capability(self.base_adapter)
        base_scope = base_capability.target_scope
        scope_matches = base_scope is None or str(base_scope) == self.target_scope
        preserve_xla = (
            bool(base_capability.is_accepted_full_chain_xla_diagnostic_authority)
            and scope_matches
        )
        nonclaims = self.nonclaims + (
            f"base value/score authority: {base_capability.value_score_authority}",
            "bootstrap latent wrapper cannot promote fallback base authority",
            "bootstrap latent wrapper preserves XLA authority only from accepted base authority",
        )
        return ValueScoreCapability(
            value_score_authority=base_capability.value_score_authority,
            xla_hmc_ready=preserve_xla,
            full_chain_xla_diagnostic_ready=preserve_xla,
            runtime_backend=self.runtime_backend,
            evidence_path=base_capability.evidence_path if preserve_xla else None,
            target_scope=self.target_scope,
            score_provenance=base_capability.score_provenance,
            nonclaims=nonclaims,
        )

    def initial_position(self) -> Any:
        import tensorflow as tf

        return tf.zeros((self.parameter_dim,), dtype=tf.float64)

    def latent_to_position(self, z: Any) -> Any:
        import tensorflow as tf

        z_tensor = self._validate_trailing_dimension(z, "latent coordinate")
        center = tf.convert_to_tensor(self.transform.center, dtype=z_tensor.dtype)
        factor = tf.convert_to_tensor(self.transform.factor, dtype=z_tensor.dtype)
        return center + tf.tensordot(z_tensor, factor, axes=[[-1], [1]])

    def theta_score_to_latent_score(self, theta_score: Any) -> Any:
        import tensorflow as tf

        score_tensor = self._validate_trailing_dimension(theta_score, "position/score")
        factor = tf.convert_to_tensor(self.transform.factor, dtype=score_tensor.dtype)
        return tf.tensordot(score_tensor, factor, axes=[[-1], [0]])

    def log_prob_and_grad(self, z: Any) -> tuple[Any, Any]:
        value, score = self._log_prob_and_grad_status(z)[:2]
        return value, score

    def _log_prob_and_grad_status(
        self, z: Any
    ) -> tuple[Any, Any, Mapping[str, Any] | None]:
        import tensorflow as tf

        z_tensor = self._validate_trailing_dimension(z, "latent coordinate")
        theta = self.latent_to_position(z_tensor)
        value_tensor, theta_score_tensor, status = (
            _call_value_score_with_batch_rank_bridge(
                self.base_adapter,
                theta,
                with_status=self.supports_retained_value_score_status,
            )
        )
        return value_tensor, self.theta_score_to_latent_score(theta_score_tensor), status

    def log_prob_and_grad_status(
        self, z: Any
    ) -> tuple[Any, Any, Mapping[str, Any]]:
        if not self.supports_retained_value_score_status:
            raise TypeError("base adapter does not expose combined value/score/status")
        value, score, status = self._log_prob_and_grad_status(z)
        if not isinstance(status, Mapping):
            raise TypeError("combined value/score/status target must return a mapping")
        return value, score, dict(status)

    def target_status_telemetry(self, z: Any) -> Mapping[str, Any]:
        telemetry = getattr(self.base_adapter, "target_status_telemetry", None)
        if not callable(telemetry):
            raise TypeError("base_adapter must expose target_status_telemetry")
        theta = self.latent_to_position(z)
        return _call_mapping_with_batch_rank_bridge(
            self.base_adapter,
            telemetry,
            theta,
        )

    def _validate_trailing_dimension(self, value: Any, label: str) -> Any:
        import tensorflow as tf

        tensor = tf.convert_to_tensor(value, dtype=tf.float64)
        if tensor.shape.rank is None:
            raise ValueError(f"{label} tensor must have static rank")
        if tensor.shape.rank < 1:
            raise ValueError(
                f"{label} tensor must have rank at least 1 with trailing parameter dimension"
            )
        trailing = tensor.shape[-1]
        if trailing is None:
            raise ValueError(f"{label} tensor must have static trailing dimension")
        if int(trailing) != self.parameter_dim:
            raise ValueError(f"{label} trailing dimension must match transform dimension")
        return tensor


def _build_bootstrap_fixed_mass_adapter(
    *,
    adapter: Any,
    mass_artifact: PrecomputedMassArtifact,
    mass_signature: str,
    target_scope: str,
    nonclaims: Sequence[str] = BOOTSTRAP_SCREEN_NONCLAIMS,
) -> _BootstrapFixedMassLatentValueScoreAdapter:
    transform = mass_artifact.build_latent_transform()
    latent_signature = program_signature(
        {
            "runtime": (
                "bayesfilter.inference.hmc_kernel_tuning."
                "_BootstrapFixedMassLatentValueScoreAdapter"
            ),
            "position_adapter_signature": stable_adapter_signature(adapter),
            "mass_artifact_signature": mass_signature,
            "target_scope": target_scope,
            "transform": transform.signature_payload(),
        }
    )
    return _BootstrapFixedMassLatentValueScoreAdapter(
        base_adapter=adapter,
        transform=transform,
        target_scope=target_scope,
        adapter_signature=latent_signature,
        nonclaims=nonclaims,
    )


def run_hmc_bootstrap_screen(
    *,
    adapter: Any,
    geometry: HMCGeometryInitializationResult,
    config: HMCBootstrapScreenConfig | None = None,
    run_full_chain: RunFullChainFn = run_full_chain_tfp_hmc,
    progress_callback: BootstrapProgressCallback | None = None,
    _private_diagnostic_callback: PrivateTuningDiagnosticCallback | None = None,
    _g2_seed_use_registry: G2PreboundarySeedUseRegistry | None = None,
) -> HMCBootstrapScreenResult:
    """Run a short fixed-kernel screen and bounded epsilon repair.

    The caller supplies a model adapter and Phase 2 geometry artifact.  The
    screen itself runs in latent fixed-mass coordinates so the recorded mass
    artifact is the operational HMC geometry.  The returned artifact supports
    later tuning phases, but is not posterior convergence or sampler-readiness
    evidence.
    """

    cfg = HMCBootstrapScreenConfig() if config is None else config
    if not isinstance(cfg, HMCBootstrapScreenConfig):
        raise TypeError("config must be HMCBootstrapScreenConfig")
    if _g2_seed_use_registry is not None and not isinstance(
        _g2_seed_use_registry,
        G2PreboundarySeedUseRegistry,
    ):
        raise TypeError("_g2_seed_use_registry must be G2PreboundarySeedUseRegistry")
    if not isinstance(geometry, HMCGeometryInitializationResult):
        raise TypeError("geometry must be HMCGeometryInitializationResult")
    adapter_signature = stable_adapter_signature(adapter)
    if adapter_signature != geometry.adapter_signature:
        raise ValueError("bootstrap adapter signature must match geometry")
    geometry.mass_artifact.validate_for_adapter(
        adapter,
        expected_dim=geometry.target_dimension,
    )
    mass_signature = _mass_artifact_signature(geometry.mass_artifact)
    if mass_signature != geometry.mass_artifact_signature:
        raise ValueError("geometry mass artifact signature mismatch")
    target_scope = _resolve_bootstrap_target_scope(adapter, cfg)
    hmc_adapter = _build_bootstrap_fixed_mass_adapter(
        adapter=adapter,
        mass_artifact=geometry.mass_artifact,
        mass_signature=mass_signature,
        target_scope=target_scope,
    )
    hmc_adapter_signature = stable_adapter_signature(hmc_adapter)
    target_dimension = int(getattr(hmc_adapter, "parameter_dim", geometry.target_dimension))
    root_seed = cfg.seed
    geometry_seed = _seed_from_mapping(geometry.seed_report, "geometry_seed")
    rounds: list[HMCBootstrapRepairRound] = []
    step = float(geometry.initial_step_size)
    target_trajectory = float(geometry.target_trajectory_length)
    low_acceptance_step: float | None = None
    high_acceptance_step: float | None = None
    unsafe_proposal_step: float | None = None
    previous_clamp_direction: str | None = None
    final_status = "repair_budget_exhausted"
    selected_index: int | None = None
    runner_cache: dict[str, Any] = {}
    runner_contract_payloads: dict[str, Mapping[str, Any]] = {}
    runner_route_events: list[Mapping[str, Any]] = []
    single_use_build_count = 0
    injected_runner_call_count = 0
    reusable_runner_build_count = 0
    use_bootstrap_reusable_route = (
        run_full_chain is run_full_chain_tfp_hmc
        and cfg.chain_execution_mode == "tf_function"
    )

    def emit_bootstrap_progress(stage: str, **payload: Any) -> None:
        if progress_callback is None:
            return
        public_payload = {
            "schema": "bayesfilter.hmc_bootstrap_public_progress.v1",
            "stage": stage,
            "round_index": int(payload["round_index"]),
            "bootstrap_diagnostic_screen_num_results": int(
                payload.get("screen_num_results", cfg.screen_num_results)
            ),
            "bootstrap_diagnostic_screen_num_burnin_steps": int(
                payload.get("screen_num_burnin_steps", cfg.screen_num_burnin_steps)
            ),
            "chain_execution_mode": cfg.chain_execution_mode,
            "use_xla": bool(cfg.use_xla),
            "target_scope": target_scope,
            "route_category": payload.get("route_category"),
            "runner_reused": payload.get("runner_reused"),
            "elapsed_s": _scalar_or_none(payload.get("elapsed_s")),
            "classification": payload.get("classification"),
            "diagnostic_role": payload.get("diagnostic_role"),
            "hard_veto_categories": tuple(payload.get("hard_veto_categories", ())),
            "acceptance_relation_to_band": payload.get("acceptance_relation_to_band"),
            "runtime_finite": payload.get("runtime_finite"),
            "round_timing_available": payload.get("round_timing_available"),
            "hmc_mechanics_exposed": False,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "reports_default_readiness": False,
            "reports_external_client_scientific_claim": False,
            "reports_gpu_or_xla_readiness": False,
            "nonclaims": BOOTSTRAP_SCREEN_NONCLAIMS,
        }
        progress_callback(stage, public_payload)

    for round_index in range(cfg.max_repairs + 1):
        leapfrog_payload = _bootstrap_leapfrog_payload(
            step,
            target_trajectory,
            max_leapfrog_steps=cfg.max_leapfrog_steps,
        )
        screen_seed = _round_seed(root_seed, round_index)
        if _g2_seed_use_registry is not None:
            try:
                screen_seed = _g2_seed_use_registry.consume(
                    derivation_site_id=(
                        _G2_BOOTSTRAP_ROUND_SEED_DERIVATION_SITE_ID
                    ),
                    terminal_gate_site_id=_G2_BOOTSTRAP_ROUND_SEED_GATE_SITE_ID,
                    key=f"bootstrap/round/{round_index:02d}",
                    owner_file="hmc_bootstrap.py",
                    owner_qualname="run_hmc_bootstrap_screen",
                    terminal_consumer="hmc_runner_interface",
                    derivation={
                        "kind": "round_offset",
                        "base_key": "bootstrap/root",
                        "round_index": round_index,
                    },
                    indices=(
                        {"name": "round_index", "value": round_index},
                    ),
                    seed=screen_seed,
                    interface_hop_site_ids=(
                        _G2_BOOTSTRAP_ROUND_SEED_INTERFACE_HOPS
                    ),
                )
            except _G2SeedRegistryError as exc:
                raise g2_preboundary_shared_invalidity_exception(
                    _g2_seed_use_registry,
                    stage=_G2_BOOTSTRAP_ROUND_SEED_GATE_SITE_ID,
                    cause=exc,
                ) from None
        screen_config = _bootstrap_screen_config(
            cfg,
            seed=screen_seed,
            step=step,
            leapfrogs=leapfrog_payload["num_leapfrog_steps"],
            target_scope=target_scope,
        )
        route_category = (
            "bootstrap_scoped_reusable_runner"
            if use_bootstrap_reusable_route
            else "single_use_or_injected_runner"
        )
        emit_bootstrap_progress(
            "bootstrap_round_start",
            round_index=round_index,
            route_category=route_category,
        )
        diagnostics: Mapping[str, Any]
        screen_error: Exception | None = None
        contract_hash: str | None = None
        route_event_recorded = False
        round_started = time.perf_counter()
        try:
            if use_bootstrap_reusable_route:
                contract_payload = _bootstrap_reusable_static_contract_payload(
                    screen_config,
                    hmc_adapter_signature=hmc_adapter_signature,
                    target_dimension=target_dimension,
                    mass_signature=mass_signature,
                )
                contract_hash = stable_config_hash(contract_payload)
                reusable_runner = runner_cache.get(contract_hash)
                runner_reused = reusable_runner is not None
                if reusable_runner is None:
                    reusable_runner = build_reusable_full_chain_tfp_hmc_runner(
                        hmc_adapter,
                        hmc_adapter.initial_position(),
                        screen_config,
                    )
                    runner_cache[contract_hash] = reusable_runner
                    runner_contract_payloads[contract_hash] = contract_payload
                    reusable_runner_build_count += 1
                emit_bootstrap_progress(
                    "bootstrap_round_hmc_call_start",
                    round_index=round_index,
                    route_category="bootstrap_scoped_reusable_runner",
                    runner_reused=runner_reused,
                    elapsed_s=time.perf_counter() - round_started,
                )
                run_result = reusable_runner.run(
                    current_state=hmc_adapter.initial_position(),
                    seed=screen_config.seed,
                    step_size=screen_config.step_size,
                )
                emit_bootstrap_progress(
                    "bootstrap_round_hmc_call_complete",
                    round_index=round_index,
                    route_category="bootstrap_scoped_reusable_runner",
                    runner_reused=runner_reused,
                    elapsed_s=time.perf_counter() - round_started,
                    runtime_finite=True,
                    round_timing_available=True,
                )
                runner_route_events.append(
                    {
                        "round_index": round_index,
                        "route": "bootstrap_scoped_reusable_runner",
                        "static_contract_hash": contract_hash,
                        "runner_reused": runner_reused,
                        "used_single_use_runner": False,
                    }
                )
                route_event_recorded = True
            else:
                used_standard_single_use = run_full_chain is run_full_chain_tfp_hmc
                if used_standard_single_use:
                    single_use_build_count += 1
                else:
                    injected_runner_call_count += 1
                emit_bootstrap_progress(
                    "bootstrap_round_hmc_call_start",
                    round_index=round_index,
                    route_category="single_use_or_injected_runner",
                    runner_reused=False,
                    elapsed_s=time.perf_counter() - round_started,
                )
                run_result = run_full_chain(
                    hmc_adapter,
                    hmc_adapter.initial_position(),
                    screen_config,
                )
                emit_bootstrap_progress(
                    "bootstrap_round_hmc_call_complete",
                    round_index=round_index,
                    route_category="single_use_or_injected_runner",
                    runner_reused=False,
                    elapsed_s=time.perf_counter() - round_started,
                    runtime_finite=True,
                    round_timing_available=True,
                )
                runner_route_events.append(
                    {
                        "round_index": round_index,
                        "route": "single_use_or_injected_runner",
                        "static_contract_hash": None,
                        "runner_reused": False,
                        "used_single_use_runner": used_standard_single_use,
                    }
                )
                route_event_recorded = True
            diagnostics = _bootstrap_diagnostics_payload(
                run_result,
                use_xla_requested=screen_config.use_xla,
                compile_chain_with_xla=screen_config.use_xla,
                expected_num_results=screen_config.num_results,
            )
        except Exception as exc:  # noqa: BLE001 - return fail-closed artifact.
            from bayesfilter.inference.hmc_preparation import HMCPreparationBudgetExceeded, HMCPreparationFailure
            if isinstance(exc, (HMCPreparationBudgetExceeded, HMCPreparationFailure)):
                raise
            screen_error = exc
            emit_bootstrap_progress(
                "bootstrap_round_hmc_call_error",
                round_index=round_index,
                route_category=route_category,
                runner_reused=False,
                elapsed_s=time.perf_counter() - round_started,
                runtime_finite=False,
                round_timing_available=True,
            )
            if not route_event_recorded:
                runner_route_events.append(
                    {
                        "round_index": round_index,
                        "route": (
                            "bootstrap_scoped_reusable_runner"
                            if use_bootstrap_reusable_route
                            else "single_use_or_injected_runner"
                        ),
                        "static_contract_hash": None,
                        "runner_reused": False,
                        "used_single_use_runner": False,
                        "failed_before_route_completion": True,
                    }
                )
            diagnostics = _bootstrap_error_diagnostics(exc)
            from bayesfilter.inference.fixed_l_finite_bracket import classify_tuning_exception
            diagnostics["failure_class"] = classify_tuning_exception(adapter, exc)
            # Capturing a failure makes a reusable runner terminal. Keep its
            # evidence, and construct a fresh runner if a retry is admissible.
            if contract_hash is not None:
                runner_cache.pop(contract_hash, None)
            if diagnostics.get("first_failure") is not None and _private_diagnostic_callback is not None:
                try:
                    _private_diagnostic_callback(
                        "first_hmc_failure",
                        {"stage": "bootstrap", "round_index": round_index,
                         "first_failure": diagnostics["first_failure"]},
                    )
                except Exception as artifact_error:
                    diagnostics["first_failure"]["artifact_write_error"] = repr(artifact_error)
        (
            classification,
            diagnostic_role,
            hard_vetoes,
            repair_triggers,
        ) = _classify_bootstrap_screen(
            cfg,
            diagnostics=diagnostics,
            screen_error=screen_error,
        )
        proposal_retry = _bootstrap_proposal_failure_is_repairable(
            diagnostics, screen_error=screen_error
        )
        if proposal_retry:
            unsafe_proposal_step = (step if unsafe_proposal_step is None
                                    else min(step, unsafe_proposal_step))
            # These remain measured acceptance endpoints, never invented
            # acceptance observations for a failed trial.
            if high_acceptance_step is not None and high_acceptance_step >= unsafe_proposal_step:
                high_acceptance_step = None
            if low_acceptance_step is not None and low_acceptance_step >= unsafe_proposal_step:
                low_acceptance_step = None
            classification = "repair"
            diagnostic_role = "declared_proposal_domain_failure_repair_only"
            repair_triggers = ("declared_proposal_domain_failure",)
        diagnostics["proposal_domain_retry_eligible"] = proposal_retry
        diagnostics["unsafe_proposal_step_bound"] = unsafe_proposal_step
        emit_bootstrap_progress(
            "bootstrap_round_classified",
            round_index=round_index,
            route_category=route_category,
            elapsed_s=time.perf_counter() - round_started,
            classification=classification,
            diagnostic_role=diagnostic_role,
            hard_veto_categories=tuple(
                dict.fromkeys(
                    _public_bootstrap_hard_veto_category(veto)
                    for veto in hard_vetoes
                )
            ),
            acceptance_relation_to_band=_bootstrap_acceptance_relation(
                diagnostics.get("mean_acceptance_probability"),
                cfg.acceptance_band,
            ),
            runtime_finite=diagnostics.get("runtime_finite"),
            round_timing_available=(
                _scalar_or_none(diagnostics.get("runtime_s")) is not None
            ),
        )
        repair_action: str | None = None
        if classification == "repair":
            acceptance = _scalar_or_none(diagnostics.get("mean_acceptance_probability"))
            if proposal_retry:
                repair_action = "smaller_epsilon_after_declared_proposal_failure_recompute_l"
            else:
                low_acceptance_step, high_acceptance_step = _bootstrap_update_repair_bracket(
                    cfg,
                    current_step=step,
                    acceptance=acceptance,
                    low_acceptance_step=low_acceptance_step,
                    high_acceptance_step=high_acceptance_step,
                )
                repair_action = _bootstrap_repair_action(
                    cfg,
                    acceptance,
                    low_acceptance_step=low_acceptance_step,
                    high_acceptance_step=high_acceptance_step,
                )
            repair_triggers = repair_triggers + (repair_action,)
        round_result = HMCBootstrapRepairRound(
            round_index=round_index,
            seed=screen_seed,
            step_size=step,
            num_leapfrog_steps=int(leapfrog_payload["num_leapfrog_steps"]),
            unclamped_num_leapfrog_steps=int(
                leapfrog_payload["unclamped_num_leapfrog_steps"]
            ),
            target_trajectory_length=target_trajectory,
            leapfrog_clamped=bool(leapfrog_payload["leapfrog_clamped"]),
            clamp_direction=leapfrog_payload["clamp_direction"],
            classification=classification,
            diagnostic_role=diagnostic_role,
            screen_config_payload=screen_config.signature_payload(),
            diagnostics=diagnostics,
            repair_action=repair_action,
            hard_vetoes=hard_vetoes,
            repair_triggers=repair_triggers,
        )
        rounds.append(round_result)
        if classification == "passed":
            selected_index = round_index
            final_status = "passed"
            break
        if classification == "hard_veto":
            final_status = "hard_veto"
            break
        if classification != "repair":
            final_status = classification
            break
        if round_index >= cfg.max_repairs:
            final_status = "repair_budget_exhausted"
            break
        try:
            if proposal_retry:
                repaired_step = _bootstrap_smaller_trial_step(
                    cfg, unsafe_step=unsafe_proposal_step,
                    finite_parent_step=high_acceptance_step,
                )
            else:
                repaired_step = _repair_step_size(
                    cfg,
                    current_step=step,
                    acceptance=acceptance,
                    low_acceptance_step=low_acceptance_step,
                    high_acceptance_step=high_acceptance_step,
                )
                if unsafe_proposal_step is not None and repaired_step >= unsafe_proposal_step:
                    repaired_step = _bootstrap_smaller_trial_step(
                        cfg, unsafe_step=unsafe_proposal_step, finite_parent_step=step,
                    )
        except ValueError as exc:
            # Preserve every completed/failed trial even at floating-point
            # search exhaustion; no unexecuted pair becomes a handoff.
            rounds[-1].diagnostics["repair_failure"] = str(exc)
            final_status = "repair_step_unavailable"
            break
        repaired_leapfrog_payload = _bootstrap_leapfrog_payload(
            repaired_step,
            target_trajectory,
            max_leapfrog_steps=cfg.max_leapfrog_steps,
        )
        if _private_diagnostic_callback is not None:
            _private_diagnostic_callback(
                "bootstrap_kernel_repair",
                {
                    "stage": "bootstrap_repair",
                    "round_index": int(round_index),
                    "previous_step_size": step,
                    "step_size": repaired_step,
                    "previous_num_leapfrog_steps": int(
                        leapfrog_payload["num_leapfrog_steps"]
                    ),
                    "num_leapfrog_steps": int(
                        repaired_leapfrog_payload["num_leapfrog_steps"]
                    ),
                    "previous_unclamped_num_leapfrog_steps": int(
                        leapfrog_payload["unclamped_num_leapfrog_steps"]
                    ),
                    "unclamped_num_leapfrog_steps": int(
                        repaired_leapfrog_payload["unclamped_num_leapfrog_steps"]
                    ),
                    "target_trajectory_length": target_trajectory,
                    "repair_action": repair_action,
                    "classification": classification,
                    "diagnostic_role": diagnostic_role,
                    "acceptance_relation_to_band": _bootstrap_acceptance_relation(
                        diagnostics.get("mean_acceptance_probability"),
                        cfg.acceptance_band,
                    ),
                    "step_size_changed": bool(repaired_step != step),
                    "num_leapfrog_steps_changed": bool(
                        int(repaired_leapfrog_payload["num_leapfrog_steps"])
                        != int(leapfrog_payload["num_leapfrog_steps"])
                    ),
                    "private_hmc_mechanics": True,
                    "reports_posterior_convergence": False,
                    "reports_sampler_superiority": False,
                    "nonclaims": BOOTSTRAP_SCREEN_NONCLAIMS,
                },
            )
        current_clamp_direction = round_result.clamp_direction
        if (
            current_clamp_direction is not None
            and current_clamp_direction == previous_clamp_direction
            and not _bootstrap_repair_makes_effective_progress(
                current_step=step,
                repaired_step=repaired_step,
                current_leapfrog_payload=leapfrog_payload,
                target_trajectory=target_trajectory,
                max_leapfrog_steps=cfg.max_leapfrog_steps,
            )
        ):
            final_status = "blocked_repeated_leapfrog_cap_saturation"
            break
        previous_clamp_direction = current_clamp_direction
        step = repaired_step

    selected_round = None if selected_index is None else rounds[selected_index]
    expected_selected_payload = (
        None
        if selected_round is None
        else _bootstrap_selected_kernel_payload(
            config=cfg,
            selected=selected_round,
            adapter_signature=adapter_signature,
            hmc_adapter_signature=hmc_adapter_signature,
            mass_artifact_signature=mass_signature,
            geometry_artifact_hash=geometry.artifact_hash,
            nonclaims=BOOTSTRAP_SCREEN_NONCLAIMS,
        )
    )
    expected_selected_hash = (
        None
        if expected_selected_payload is None
        else stable_config_hash(expected_selected_payload)
    )
    bootstrap_runner_route = {
        "active_route": (
            "bootstrap_scoped_reusable_runner"
            if use_bootstrap_reusable_route
            else "single_use_or_injected_runner"
        ),
        "semantic_source": "run_hmc_bootstrap_screen",
        "single_use_build_count_for_bootstrap_rounds": single_use_build_count,
        "injected_runner_call_count": injected_runner_call_count,
        "reusable_runner_build_count": reusable_runner_build_count,
        "distinct_static_runner_contract_count": len(runner_contract_payloads),
        "distinct_static_runner_contracts": tuple(
            {
                "static_contract_hash": contract_hash,
                "static_contract_payload": runner_contract_payloads[contract_hash],
            }
            for contract_hash in sorted(runner_contract_payloads)
        ),
        "bootstrap_round_count": len(rounds),
        "fallback_to_single_use_runner": (
            use_bootstrap_reusable_route and single_use_build_count > 0
        ),
        "fallback_status": (
            "none"
            if use_bootstrap_reusable_route and single_use_build_count == 0
            else "inactive_reusable_route"
        ),
        "round_route_events": tuple(runner_route_events),
        "selected_kernel_payload_hash": expected_selected_hash,
        "selected_kernel_preservation": {
            "checked": selected_round is not None,
            "semantic_source": "run_hmc_bootstrap_screen",
            "payload_hash": expected_selected_hash,
            "selected_round_index": selected_index,
            "preserved": selected_round is not None,
            "step_size": None if selected_round is None else selected_round.step_size,
            "num_leapfrog_steps": (
                None if selected_round is None else selected_round.num_leapfrog_steps
            ),
            "seed": None if selected_round is None else selected_round.seed,
            "screen_config_payload": (
                None if selected_round is None else selected_round.screen_config_payload
            ),
        },
        "dynamic_runner_inputs": ("current_state", "seed", "step_size"),
        "dynamic_inputs_preserve_round_semantics": True,
    }

    return HMCBootstrapScreenResult(
        config=cfg,
        geometry_artifact_hash=geometry.artifact_hash,
        adapter_signature=adapter_signature,
        hmc_adapter_signature=hmc_adapter_signature,
        mass_artifact_signature=mass_signature,
        target_dimension=target_dimension,
        rounds=tuple(rounds),
        selected_round_index=selected_index,
        final_status=final_status,
        seed_report={
            "geometry_root_seed": geometry.seed_report.get("root_seed"),
            "geometry_seed": geometry_seed,
            "bootstrap_root_seed": root_seed,
            "bootstrap_stage_seed": _derive_seed(root_seed, stage_index=0),
            "screen_round_seeds": tuple(round_result.seed for round_result in rounds),
            "seed_owner": "BayesFilter",
            "geometry_seed_distinct_from_bootstrap_seed": geometry_seed != root_seed,
        },
        diagnostic_roles={
            "mean_acceptance_probability": "bootstrap_decision_and_repair",
            "binary_acceptance_rate": "explanatory_movement_diagnostic",
            "declared_proposal_domain_failure": "failed_trial_veto_and_bounded_repair_trigger",
            "acceptance_band": ("startup_floor_only" if cfg.acceptance_role == "warmup_startup_only"
                                else "bootstrap_promotion_only"),
            "repair_band": "bootstrap_repair_trigger",
            "finite_samples": "hard_veto",
            "log_accept_ratio_finite": "hard_veto",
            "target_log_prob_finite": "hard_veto",
            "adapter_signature_match": "hard_veto",
            "mass_artifact_validity": "hard_veto",
            "repeated_leapfrog_cap_saturation": "phase_blocker_handoff",
            "runtime": "explanatory_diagnostic",
        },
        bootstrap_runner_route=bootstrap_runner_route,
    )


def _bootstrap_selected_kernel_payload(
    *,
    config: "HMCBootstrapScreenConfig",
    selected: "HMCBootstrapRepairRound",
    adapter_signature: str,
    hmc_adapter_signature: str,
    mass_artifact_signature: str,
    geometry_artifact_hash: str,
    nonclaims: tuple[str, ...],
) -> Mapping[str, Any]:
    return {
        "runtime": "bayesfilter.inference.run_hmc_bootstrap_screen",
        "sample_space": "latent_fixed_mass",
        "step_size": selected.step_size,
        "num_leapfrog_steps": selected.num_leapfrog_steps,
        "target_trajectory_length": selected.target_trajectory_length,
        "target_accept_prob": config.target_accept_prob,
        "acceptance_role": config.acceptance_role,
        "acceptance_statistic": BOOTSTRAP_ACCEPTANCE_STATISTIC,
        "startup_acceptance_floor": (config.repair_band[0]
                                     if config.acceptance_role == "warmup_startup_only" else None),
        "acceptance_band": config.acceptance_band,
        "repair_band": config.repair_band,
        "adapter_signature": adapter_signature,
        "hmc_adapter_signature": hmc_adapter_signature,
        "mass_artifact_signature": mass_artifact_signature,
        "geometry_artifact_hash": geometry_artifact_hash,
        "seed": selected.seed,
        "screen_config_payload": selected.screen_config_payload,
        "nonclaims": nonclaims,
    }


def _bootstrap_acceptance_relation(
    value: Any,
    acceptance_band: tuple[float, float],
) -> str:
    acceptance = _scalar_or_none(value)
    if acceptance is None or not math.isfinite(acceptance):
        return "missing_or_nonfinite"
    lower, upper = acceptance_band
    if acceptance < lower:
        return "below"
    if acceptance > upper:
        return "above"
    return "inside"


def _bootstrap_diagnostics_payload(
    run_result: FullChainHMCRunResult,
    *,
    use_xla_requested: bool | None = None,
    compile_chain_with_xla: bool | None = None,
    expected_num_results: int | None = None,
) -> Mapping[str, Any]:
    """Summarize actual runner tensors before constructing host-side screen metadata.

    Finite masks and extrema stay in TensorFlow. Nonempty sample and target
    evidence is required: an empty reduction or absent target trace must never
    make a failed/incomplete runner look healthy to candidate selection.
    """

    import tensorflow as tf

    diagnostics = dict(run_result.diagnostics)
    trace = dict(run_result.trace)
    metadata = dict(run_result.metadata)
    acceptance = _scalar_or_none(diagnostics.get("acceptance_rate"))
    runtime_s = _runtime_seconds_or_none(metadata)
    use_xla_metadata = _bool_or_none(metadata.get("use_xla"))
    jit_compile_metadata = _bool_or_none(metadata.get("jit_compile"))
    use_xla = (
        bool(use_xla_requested)
        if use_xla_requested is not None
        else bool(use_xla_metadata)
    )
    jit_compile = (
        bool(compile_chain_with_xla)
        if compile_chain_with_xla is not None
        else bool(jit_compile_metadata)
    )
    payload: dict[str, Any] = {
        # Keep the historical runner field binary. Decisions below use only
        # the explicitly named probability computed from the proposal trace.
        "acceptance_rate": acceptance,
        "binary_acceptance_rate": acceptance,
        "mean_acceptance_probability": None,
        "acceptance_statistic": BOOTSTRAP_ACCEPTANCE_STATISTIC,
        "acceptance_probability_count": 0,
        "acceptance_probability_unavailable_reason": "missing_log_accept_ratio",
        "runtime_s": runtime_s,
        "runtime_finite": runtime_s is not None and math.isfinite(runtime_s),
        "use_xla": use_xla,
        "xla_requested": use_xla,
        "compile_chain_with_xla": jit_compile,
        "jit_compile_metadata": "true" if jit_compile else "false",
        "finite_sample_count": _int_or_none(diagnostics.get("finite_sample_count")),
        "nonfinite_sample_count": _int_or_none(
            diagnostics.get("nonfinite_sample_count")
        ),
        "target_accept_prob": _scalar_or_none(diagnostics.get("target_accept_prob")),
        "num_adaptation_steps": _int_or_none(diagnostics.get("num_adaptation_steps")),
        "trace_policy": diagnostics.get("trace_policy"),
        "divergence_status": diagnostics.get("divergence_status"),
        "divergence_count": _int_or_none(diagnostics.get("divergence_count")),
        "target_status_telemetry": _telemetry_payload(
            diagnostics.get("target_status_telemetry")
        ),
        "log_accept_ratio_finite": None,
        "max_abs_log_accept_ratio": None,
        "target_log_prob_finite": None,
        "samples_all_finite": None,
        "screen_diagnostic": None,
        "raw_diagnostics": _json_ready(diagnostics),
        "runtime_metadata": _json_ready(metadata),
        "trace_summary": {
            "trace_keys": tuple(sorted(trace.keys())),
            "trace_unavailability": metadata.get("trace_unavailability"),
        },
    }
    log_accept = None
    if "log_accept_ratio" in trace:
        log_accept = tf.cast(
            tf.convert_to_tensor(trace["log_accept_ratio"], dtype_hint=tf.float64), tf.float64
        )
        finite = tf.math.is_finite(log_accept)
        payload["log_accept_ratio_finite"] = bool(tf.logical_and(
            tf.size(log_accept) > 0, tf.reduce_all(finite)
        ).numpy())
        finite_values = tf.boolean_mask(tf.reshape(log_accept, [-1]), tf.reshape(finite, [-1]))
        payload["max_abs_log_accept_ratio"] = (
            None if not bool(tf.reduce_any(finite).numpy())
            else float(tf.reduce_max(tf.abs(finite_values)).numpy())
        )
    target_log_prob = None
    if "target_log_prob" in trace:
        target_log_prob = tf.cast(
            tf.convert_to_tensor(trace["target_log_prob"], dtype_hint=tf.float64), tf.float64
        )
        payload["target_log_prob_finite"] = bool(tf.logical_and(
            tf.size(target_log_prob) > 0, tf.reduce_all(tf.math.is_finite(target_log_prob))
        ).numpy())
    samples = tf.cast(tf.convert_to_tensor(run_result.samples, dtype_hint=tf.float64), tf.float64)
    if log_accept is not None:
        probability_count = int(tf.size(log_accept).numpy())
        payload["acceptance_probability_count"] = probability_count
        if probability_count == 0:
            reason = "empty_log_accept_ratio"
        elif (samples.shape.rank not in (2, 3)
              or log_accept.shape != samples.shape[:-1]):
            reason = "sample_and_acceptance_shapes_differ"
        elif (expected_num_results is not None
              and int(tf.shape(log_accept)[0].numpy()) != expected_num_results):
            reason = "recorded_proposal_count_mismatch"
        elif payload["log_accept_ratio_finite"] is not True:
            reason = "nonfinite_log_accept_ratio"
        else:
            # TFP's result trace excludes burnin, and includes the proposal
            # log ratio even when MH rejects. Never filter by is_accepted.
            payload["mean_acceptance_probability"] = float(tf.reduce_mean(
                tf.exp(tf.minimum(log_accept, 0.0))
            ).numpy())
            reason = None
        payload["acceptance_probability_unavailable_reason"] = reason
    samples_all_finite = bool(tf.logical_and(
        tf.size(samples) > 0, tf.reduce_all(tf.math.is_finite(samples))
    ).numpy())
    payload["samples_all_finite"] = samples_all_finite
    required_arrays_finite = bool(
        samples_all_finite and payload["target_log_prob_finite"] is True
    )
    screen = screen_hmc_diagnostics(
        sample_chain_returned=True,
        hmc_error_absent=True,
        required_arrays_finite=required_arrays_finite,
        log_accept_ratio=log_accept,
        divergences=trace.get("divergence"),
        acceptance_rate_by_chain=None if acceptance is None else (acceptance,),
        fixed_kernel_used=True,
        num_adaptation_steps_zero=True,
        latent_initial_scale_zero=True,
        use_xla_false=None if use_xla else True,
        compile_chain_with_xla_false=None if jit_compile else True,
        diagnostic_role="bootstrap fixed-kernel screen only",
    )
    payload["screen_diagnostic"] = {
        "passed": screen.passed,
        "checks": dict(screen.checks),
        "diagnostic_roles": dict(screen.diagnostic_roles),
        "unavailable_diagnostics": screen.unavailable_diagnostics,
        "nonclaims": screen.nonclaims,
    }
    if target_log_prob is not None:
        finite = tf.boolean_mask(
            tf.reshape(target_log_prob, [-1]),
            tf.reshape(tf.math.is_finite(target_log_prob), [-1]),
        )
        has_finite = bool((tf.size(finite) > 0).numpy())
        payload["target_log_prob_min"] = (
            float(tf.reduce_min(finite).numpy()) if has_finite else None
        )
        payload["target_log_prob_max"] = (
            float(tf.reduce_max(finite).numpy()) if has_finite else None
        )
    return payload


def _bootstrap_error_diagnostics(exc: Exception) -> Mapping[str, Any]:
    message = str(exc)
    if len(message) > 2000:
        message = message[:1997] + "..."
    return {
        "error_type": type(exc).__name__,
        "error_message": message,
        **({"first_failure": getattr(exc, "failure_record")}
           if getattr(exc, "failure_record", None) is not None else {}),
        "failure_diagnostics_role": "diagnostic_exception_provenance_not_scientific_evidence",
        "acceptance_rate": None,
        "binary_acceptance_rate": None,
        "mean_acceptance_probability": None,
        "acceptance_statistic": BOOTSTRAP_ACCEPTANCE_STATISTIC,
        "acceptance_probability_count": 0,
        "acceptance_probability_unavailable_reason": "screen_execution_failed",
        "runtime_s": None,
        "runtime_finite": False,
        "samples_all_finite": False,
        "log_accept_ratio_finite": False,
        "target_log_prob_finite": False,
        "screen_diagnostic": {
            "passed": False,
            "checks": {
                "sample_chain_returned": False,
                "hmc_error_absent": False,
            },
            "diagnostic_roles": {"screen": "bootstrap runtime hard veto"},
            "unavailable_diagnostics": (),
            "nonclaims": BOOTSTRAP_SCREEN_NONCLAIMS,
        },
    }


def _bootstrap_proposal_failure_is_repairable(
    diagnostics: Mapping[str, Any], *, screen_error: Exception | None,
) -> bool:
    """Require both adapter classification and repository proposal attribution.

    Error strings and InvalidArgumentError alone cannot establish locality.
    The initial bootstrap and retained trace are never repairable here.
    """
    import tensorflow as tf

    if not isinstance(screen_error, tf.errors.InvalidArgumentError):
        return False
    record = diagnostics.get("first_failure")
    if (diagnostics.get("failure_class") != "target_domain"
            or not isinstance(record, Mapping)
            or record.get("schema") != "bayesfilter.traced_hmc_first_failure.v1"
            or record.get("evaluation_phase") != "trajectory"
            or record.get("failure_location") != "target_callback"
            or type(record.get("failed_leapfrog_substep")) is not int
            or record["failed_leapfrog_substep"] < 1):
        return False
    state = record.get("pre_transition_state")

    def finite_state(value: Any) -> bool:
        if isinstance(value, (list, tuple)):
            return bool(value) and all(finite_state(item) for item in value)
        return type(value) in (int, float) and math.isfinite(value)

    return isinstance(state, (list, tuple)) and finite_state(state)


def _bootstrap_smaller_trial_step(
    config: HMCBootstrapScreenConfig, *, unsafe_step: float,
    finite_parent_step: float | None,
) -> float:
    """Nominate an interior trial without fabricating acceptance evidence."""
    if finite_parent_step is not None and 0.0 < finite_parent_step < unsafe_step:
        candidate = math.exp(0.5 * (math.log(finite_parent_step) + math.log(unsafe_step)))
    else:
        candidate = unsafe_step / config.step_repair_factor
    if not math.isfinite(candidate) or not 0.0 < candidate < unsafe_step:
        raise ValueError("bootstrap unsafe proposal bound has no smaller finite trial")
    return candidate


def _bootstrap_leapfrog_payload(
    step: float,
    target_trajectory: float,
    *,
    max_leapfrog_steps: int = _GEOMETRY_MAX_LEAPFROG,
) -> Mapping[str, Any]:
    raw = int(math.ceil(float(target_trajectory) / float(step)))
    if raw <= 0:
        raise ValueError("formula-derived bootstrap leapfrog count must be positive")
    max_l = _validate_max_leapfrog_steps(max_leapfrog_steps)
    leapfrogs = min(max_l, max(_GEOMETRY_MIN_LEAPFROG, raw))
    if leapfrogs == raw:
        clamp_direction = None
    elif raw < _GEOMETRY_MIN_LEAPFROG:
        clamp_direction = "min"
    else:
        clamp_direction = "max"
    return {
        "num_leapfrog_steps": leapfrogs,
        "unclamped_num_leapfrog_steps": raw,
        "leapfrog_clamped": leapfrogs != raw,
        "clamp_direction": clamp_direction,
        "internal_min_leapfrog": _GEOMETRY_MIN_LEAPFROG,
        "internal_max_leapfrog": max_l,
        "default_max_leapfrog_steps": _GEOMETRY_MAX_LEAPFROG,
    }


def _bootstrap_repair_action(
    config: HMCBootstrapScreenConfig,
    acceptance: float | None,
    *,
    low_acceptance_step: float | None = None,
    high_acceptance_step: float | None = None,
) -> str:
    if acceptance is None or not math.isfinite(float(acceptance)):
        return "no_repair_nonfinite_acceptance"
    if low_acceptance_step is not None and high_acceptance_step is not None:
        _validate_bootstrap_repair_bracket(
            low_acceptance_step=low_acceptance_step,
            high_acceptance_step=high_acceptance_step,
        )
        return "bracketed_log_step_midpoint_recompute_l"
    if float(acceptance) < config.acceptance_band[0]:
        return "reduce_epsilon_recompute_l"
    return "increase_epsilon_recompute_l"


def _bootstrap_repair_makes_effective_progress(
    *,
    current_step: float,
    repaired_step: float,
    current_leapfrog_payload: Mapping[str, Any],
    target_trajectory: float,
    max_leapfrog_steps: int,
) -> bool:
    """Return whether the next repair changes the bootstrap HMC kernel."""

    candidate_payload = _bootstrap_leapfrog_payload(
        repaired_step,
        target_trajectory,
        max_leapfrog_steps=max_leapfrog_steps,
    )
    return bool(
        float(repaired_step) != float(current_step)
        or int(candidate_payload["num_leapfrog_steps"])
        != int(current_leapfrog_payload["num_leapfrog_steps"])
        or candidate_payload["clamp_direction"] != current_leapfrog_payload["clamp_direction"]
    )


def _bootstrap_reusable_static_contract_payload(
    config: FullChainHMCConfig,
    *,
    hmc_adapter_signature: str,
    target_dimension: int,
    mass_signature: str,
) -> Mapping[str, Any]:
    """Return the bootstrap fields that fix the reusable runner graph.

    ``ReusableFullChainHMCRunner`` accepts seed, current state, and step size as
    runtime tensors.  They are preserved in every round's
    ``screen_config_payload`` and selected-kernel payload, so they must not be
    part of the reusable static contract key.
    """

    tuning_payload = config.tuning_policy.payload()
    return {
        "semantic_source": "run_hmc_bootstrap_screen",
        "runner": "build_reusable_full_chain_tfp_hmc_runner",
        "hmc_adapter_signature": hmc_adapter_signature,
        "mass_artifact_signature": mass_signature,
        "target_dimension": int(target_dimension),
        "num_results": config.num_results,
        "num_burnin_steps": config.num_burnin_steps,
        **({"capture_first_failure": True, "failure_capture_role": config.failure_capture_role}
           if config.capture_first_failure else {}),
        "num_leapfrog_steps": config.num_leapfrog_steps,
        "use_xla": config.use_xla,
        "chain_execution_mode": config.chain_execution_mode,
        "trace_policy": config.trace_policy,
        "target_status_trace_policy": config.target_status_trace_policy,
        "adaptation_policy": config.adaptation_policy,
        "tuning_policy": tuning_payload,
        "target_scope": config.target_scope,
        "dynamic_inputs": ("current_state", "seed", "step_size"),
        "excluded_dynamic_fields": ("seed", "step_size"),
    }


def _bootstrap_screen_config(
    config: HMCBootstrapScreenConfig,
    *,
    seed: tuple[int, int],
    step: float,
    leapfrogs: int,
    target_scope: str,
) -> FullChainHMCConfig:
    return FullChainHMCConfig(
        num_results=config.screen_num_results,
        num_burnin_steps=config.screen_num_burnin_steps,
        step_size=float(step),
        num_leapfrog_steps=int(leapfrogs),
        seed=seed,
        use_xla=config.use_xla,
        trace_policy="standard",
        target_status_trace_policy=config.target_status_trace_policy,
        target_scope=target_scope,
        chain_execution_mode=config.chain_execution_mode,
        capture_first_failure=config.chain_execution_mode == "tf_function" and not config.use_xla,
        failure_capture_role="bootstrap_screen",
    )


def _bootstrap_update_repair_bracket(
    config: HMCBootstrapScreenConfig,
    *,
    current_step: float,
    acceptance: float | None,
    low_acceptance_step: float | None,
    high_acceptance_step: float | None,
) -> tuple[float | None, float | None]:
    if acceptance is None or not math.isfinite(float(acceptance)):
        return low_acceptance_step, high_acceptance_step
    step = float(current_step)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("bootstrap repair bracket step must be positive and finite")
    if float(acceptance) < config.acceptance_band[0]:
        low_acceptance_step = step
    else:
        high_acceptance_step = step
    if low_acceptance_step is not None and high_acceptance_step is not None:
        _validate_bootstrap_repair_bracket(
            low_acceptance_step=low_acceptance_step,
            high_acceptance_step=high_acceptance_step,
        )
    return low_acceptance_step, high_acceptance_step


def _classify_bootstrap_screen(
    config: HMCBootstrapScreenConfig,
    *,
    diagnostics: Mapping[str, Any],
    screen_error: Exception | None,
) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    hard_vetoes: list[str] = []
    if screen_error is not None:
        hard_vetoes.append("screen_hmc_error")
    acceptance = diagnostics.get("mean_acceptance_probability")
    if acceptance is None or not _finite_number(acceptance):
        hard_vetoes.append("screen_acceptance_missing_or_nonfinite")
    if diagnostics.get("runtime_finite") is not True:
        hard_vetoes.append("screen_runtime_missing_or_nonfinite")
    if diagnostics.get("log_accept_ratio_finite") is not True:
        hard_vetoes.append("screen_log_accept_nonfinite_or_missing")
    if diagnostics.get("samples_all_finite") is not True:
        hard_vetoes.append("screen_samples_nonfinite_or_missing")
    if diagnostics.get("target_log_prob_finite") is not True:
        hard_vetoes.append("screen_target_log_prob_nonfinite_or_missing")
    telemetry = diagnostics.get("target_status_telemetry")
    if isinstance(telemetry, Mapping) and telemetry.get("telemetry_failure_veto_bool"):
        hard_vetoes.append("screen_target_status_telemetry_failure")
    if hard_vetoes:
        return "hard_veto", "hard_veto", tuple(hard_vetoes), ()
    acceptance_value = float(acceptance)
    if config.acceptance_role == "warmup_startup_only":
        if acceptance_value >= config.repair_band[0]:
            return "passed", "finite_startup_for_adaptation_only", (), ()
        return "repair", "startup_acceptance_repair_trigger", (), ("acceptance_below_repair_band",)
    if config.acceptance_band[0] <= acceptance_value <= config.acceptance_band[1]:
        return "passed", "bootstrap_screen_promotion_only", (), ()
    if acceptance_value < config.acceptance_band[0]:
        trigger = (
            "acceptance_below_repair_band"
            if acceptance_value < config.repair_band[0]
            else "acceptance_below_acceptance_band_inside_repair_band"
        )
        return (
            "repair",
            "bootstrap_acceptance_repair_trigger",
            (),
            (trigger,),
        )
    trigger = (
        "acceptance_above_repair_band"
        if acceptance_value > config.repair_band[1]
        else "acceptance_above_acceptance_band_inside_repair_band"
    )
    return (
        "repair",
        "bootstrap_acceptance_repair_trigger",
        (),
        (trigger,),
    )


def _public_bootstrap_hard_veto_category(veto: str) -> str:
    mapping = {
        "screen_hmc_error": "screen_hmc_error",
        "screen_acceptance_missing_or_nonfinite": "nonfinite_acceptance",
        "screen_runtime_missing_or_nonfinite": "screen_hmc_error",
        "screen_log_accept_nonfinite_or_missing": "nonfinite_log_accept",
        "screen_samples_nonfinite_or_missing": "nonfinite_samples",
        "screen_target_log_prob_nonfinite_or_missing": "nonfinite_target_log_prob",
        "screen_target_status_telemetry_failure": "target_status_telemetry_failure",
    }
    return mapping.get(str(veto), "screen_hmc_error")


def _repair_step_size(
    config: HMCBootstrapScreenConfig,
    *,
    current_step: float,
    acceptance: float | None,
    low_acceptance_step: float | None = None,
    high_acceptance_step: float | None = None,
) -> float:
    if acceptance is None or not math.isfinite(float(acceptance)):
        raise ValueError("finite acceptance is required for epsilon repair")
    if low_acceptance_step is not None and high_acceptance_step is not None:
        _validate_bootstrap_repair_bracket(
            low_acceptance_step=low_acceptance_step,
            high_acceptance_step=high_acceptance_step,
        )
        repaired = float(math.exp(
            0.5 * (math.log(float(low_acceptance_step)) + math.log(float(high_acceptance_step)))
        ))
    else:
        factor = (
            1.0 / config.step_repair_factor
            if float(acceptance) < config.acceptance_band[0]
            else config.step_repair_factor
        )
        repaired = float(current_step) * factor
    if not math.isfinite(repaired) or repaired <= 0.0:
        raise ValueError("repaired bootstrap step size must be positive and finite")
    return repaired


def _resolve_bootstrap_target_scope(
    adapter: Any,
    config: HMCBootstrapScreenConfig,
) -> str:
    capability = value_score_capability(adapter)
    capability_scope = capability.target_scope
    if config.target_scope is not None:
        target_scope = str(config.target_scope)
        if not target_scope:
            raise ValueError("target_scope must be non-empty when provided")
        if capability_scope is not None and target_scope != capability_scope:
            raise ValueError("value/score target_scope mismatch")
        return target_scope
    if capability_scope is None or not str(capability_scope):
        raise ValueError(
            "bootstrap screen requires config.target_scope or an adapter "
            "value_score_capability target_scope"
        )
    return str(capability_scope)


def _validate_bootstrap_repair_bracket(
    *,
    low_acceptance_step: float,
    high_acceptance_step: float,
) -> None:
    low_step = float(low_acceptance_step)
    high_step = float(high_acceptance_step)
    if (
        not math.isfinite(low_step)
        or low_step <= 0.0
        or not math.isfinite(high_step)
        or high_step <= 0.0
    ):
        raise ValueError("bootstrap repair bracket endpoints must be positive and finite")
    if not math.log(high_step) < math.log(low_step):
        raise ValueError("bootstrap repair bracket endpoints must be ordered in log-step space")


BootstrapFixedMassAdapter = _BootstrapFixedMassLatentValueScoreAdapter
build_bootstrap_fixed_mass_adapter = _build_bootstrap_fixed_mass_adapter

__all__ = [
    "BOOTSTRAP_SCREEN_NONCLAIMS", "BootstrapFixedMassAdapter",
    "HMCBootstrapRepairRound", "HMCBootstrapScreenConfig", "HMCBootstrapScreenResult",
    "build_bootstrap_fixed_mass_adapter", "run_hmc_bootstrap_screen",
]
