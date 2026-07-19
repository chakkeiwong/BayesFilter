"""Pure HMC algorithm-route identity and capability contracts.

This module deliberately has no TensorFlow, TFP, or inference-package imports.
It resolves an algorithm that the caller has already selected; execution
controls may affect support or closeout behavior, but never choose a different
algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


HMC_ROUTE_CONTRACT_VERSION = "bayesfilter.hmc_algorithm_route.v1"

OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID = (
    "operational_interleaved_windowed_warmup_v2"
)
OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID = (
    "operational_paired_fixed_trajectory_selection_v3"
)
LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID = "segmented_windowed_mass_runner"
LEGACY_JOINT_L_EPSILON_ALGORITHM_ID = "joint_l_epsilon_grid_fixed_mass_hmc"

HMC_WINDOWED_MASS_STAGE = "windowed_mass"
HMC_FIXED_TRAJECTORY_STAGE = "fixed_trajectory"
HMC_TOP_LEVEL_SELECTION_STAGE = "top_level_selection"

_KNOWN_BY_STAGE = {
    HMC_WINDOWED_MASS_STAGE: {
        OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
        LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID,
    },
    HMC_FIXED_TRAJECTORY_STAGE: {
        OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID,
        LEGACY_JOINT_L_EPSILON_ALGORITHM_ID,
    },
    HMC_TOP_LEVEL_SELECTION_STAGE: {
        OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID,
        LEGACY_JOINT_L_EPSILON_ALGORITHM_ID,
    },
}


@dataclass(frozen=True)
class HMCAlgorithmRouteDecision:
    """Immutable decision about one explicitly selected HMC algorithm."""

    algorithm_id: str
    stage: str
    route_contract_version: str
    supported: bool
    blocker_code: str | None
    runtime_backend_requirement: str
    xla_requirement: str
    timeout_capability: str
    heartbeat_capability: str
    output_path_capability: str
    checkpointing_capability: str
    runner_requirement: str
    evidence_role: str
    promotion_role: str
    stopping_rule_role: str
    operational_authority: bool
    reports_posterior_convergence: bool = False
    reports_sampler_superiority: bool = False

    def payload(self) -> Mapping[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "stage": self.stage,
            "route_contract_version": self.route_contract_version,
            "supported": self.supported,
            "blocker_code": self.blocker_code,
            "runtime_backend_requirement": self.runtime_backend_requirement,
            "xla_requirement": self.xla_requirement,
            "execution_control_capabilities": {
                "timeout": self.timeout_capability,
                "heartbeat": self.heartbeat_capability,
                "output_path": self.output_path_capability,
                "checkpointing": self.checkpointing_capability,
            },
            "runner_requirement": self.runner_requirement,
            "evidence_role": self.evidence_role,
            "promotion_role": self.promotion_role,
            "stopping_rule_role": self.stopping_rule_role,
            "operational_authority": self.operational_authority,
            "reports_posterior_convergence": self.reports_posterior_convergence,
            "reports_sampler_superiority": self.reports_sampler_superiority,
        }


class UnsupportedHMCAlgorithmRoute(RuntimeError):
    """Raised before runtime construction when a route decision is blocked."""

    def __init__(self, decision: HMCAlgorithmRouteDecision) -> None:
        if decision.supported or decision.blocker_code is None:
            raise ValueError("route exception requires an unsupported decision")
        self.decision = decision
        super().__init__(
            f"HMC algorithm route blocked: {decision.blocker_code} "
            f"({decision.stage}: {decision.algorithm_id})"
        )


def windowed_algorithm_for_selection_algorithm(algorithm_id: str) -> str:
    """Map a top-level selection family to its compatible warmup algorithm."""

    selected = str(algorithm_id)
    if selected == OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID:
        return OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
    if selected == LEGACY_JOINT_L_EPSILON_ALGORITHM_ID:
        return LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID
    raise ValueError(f"unknown top-level HMC selection algorithm_id: {selected}")


def resolve_hmc_algorithm_route(
    *,
    algorithm_id: str,
    stage: str,
    runtime_backend: str = "tensorflow",
    chain_execution_mode: str = "tf_function",
    use_xla: bool = False,
    timeout_enabled: bool = False,
    heartbeat_enabled: bool = False,
    output_path_enabled: bool = False,
    checkpointing_enabled: bool = False,
    runner_identity: str = "default",
) -> HMCAlgorithmRouteDecision:
    """Resolve support without letting execution controls select the route."""

    selected = str(algorithm_id)
    stage_name = str(stage)
    backend = str(runtime_backend)
    mode = str(chain_execution_mode)
    runner = str(runner_identity)
    known = _KNOWN_BY_STAGE.get(stage_name, set())

    blocker: str | None = None
    if stage_name not in _KNOWN_BY_STAGE:
        blocker = "unsupported_stage"
    elif selected not in known:
        blocker = "unknown_or_stage_incompatible_algorithm_id"
    elif backend != "tensorflow":
        blocker = "unsupported_runtime_backend"
    elif mode not in {"eager", "tf_function"}:
        blocker = "unsupported_chain_execution_mode"
    elif use_xla and mode != "tf_function":
        blocker = "xla_requires_tf_function"
    elif selected == OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID and use_xla:
        blocker = "operational_windowed_warmup_xla_not_validated"
    elif (
        selected == OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
        and runner != "default"
    ):
        blocker = "operational_windowed_warmup_requires_default_runner"

    operational = selected in {
        OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
        OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID,
    }
    legacy = selected in {
        LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID,
        LEGACY_JOINT_L_EPSILON_ALGORITHM_ID,
    }
    timeout_capability = (
        "window_boundary_closeout"
        if selected == OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
        else "legacy_segment_boundary_closeout"
        if selected == LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID
        else "stage_boundary_closeout"
    )
    # These values are deliberately computed independently of whether controls
    # are enabled. Referencing the booleans documents that they were considered
    # while making it impossible for them to alter algorithm identity.
    _ = (timeout_enabled, heartbeat_enabled, output_path_enabled, checkpointing_enabled)
    return HMCAlgorithmRouteDecision(
        algorithm_id=selected,
        stage=stage_name,
        route_contract_version=HMC_ROUTE_CONTRACT_VERSION,
        supported=blocker is None,
        blocker_code=blocker,
        runtime_backend_requirement="tensorflow_probability",
        xla_requirement=(
            "not_validated"
            if selected == OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
            else "caller_configured_tf_function_only"
        ),
        timeout_capability=timeout_capability,
        heartbeat_capability="telemetry_only",
        output_path_capability="artifact_only",
        checkpointing_capability="boundary_artifact_only",
        runner_requirement=(
            "default_operational_runner"
            if selected == OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
            else "explicit_legacy_or_test_runner"
            if legacy
            else "stage_policy_runner"
        ),
        evidence_role="engineering_only",
        promotion_role="non_promoting" if legacy else "stage_handoff_only",
        stopping_rule_role="not_a_stopping_rule",
        operational_authority=bool(operational and blocker is None),
    )


def require_hmc_algorithm_route(**kwargs: Any) -> HMCAlgorithmRouteDecision:
    """Return a supported route or raise its typed blocker."""

    decision = resolve_hmc_algorithm_route(**kwargs)
    if not decision.supported:
        raise UnsupportedHMCAlgorithmRoute(decision)
    return decision
