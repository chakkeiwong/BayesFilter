"""Shared state-continuing epsilon-repair broad-grid tuning for NeuTra targets.

This is the reviewed PP-UKF repaired tuning procedure extracted from
``docs/benchmarks/run_pp_ukf_state_continuing_epsilon_repair_20260721.py`` into
reusable inference code. Mechanics are preserved:

- per-``L`` dual-averaging adaptation from a warm-start epsilon, with a frozen
  common epsilon verified over the post-adaptation window;
- bounded, bracketed epsilon repair toward a declared acceptance calibration
  region, continuing the chain state across repair steps;
- fresh final screens launched from the calibrated per-``L`` state;
- same-epsilon one-hop coverage probes launched from the calibrated parent
  state, never retuned;
- classification, barriers, and the unranked next-round union delegate to the
  reviewed operational broad-grid contract.

All draws are discarded tuning evidence. This module never retains samples and
never issues a posterior, convergence, ranking, or default-readiness claim.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import tensorflow as tf

from bayesfilter.inference.hmc import (
    FullChainHMCConfig,
    build_reusable_full_chain_tfp_hmc_runner,
)
from bayesfilter.inference.hmc_operational_broad_grid import (
    PRIMARY_L_GRID,
    OperationalBroadGridPolicy,
    OperationalMassHandoff,
    OperationalPrimaryCandidate,
    SameEpsilonNeighborGuard,
    classify_operational_pair_evidence,
    operational_broad_seed,
    run_operational_broad_grid,
)
from bayesfilter.inference.hmc_tuning import HMCTuningPolicy
from bayesfilter.inference.neutra_broad_grid import (
    CHAIN_COUNT,
    DEFAULT_REQUIRED_STATUS_KEYS,
    REPLICATION_COUNT,
    _json_ready,
    _normalized_status_keys,
    _signature,
    build_fixed_identity_broad_grid_handoff,
    evaluate_fixed_screen_run,
)
from bayesfilter.runtime import atomic_write_json


DEFAULT_ADAPTATION_STEPS = 96
DEFAULT_POST_ADAPTATION_RESULTS = 32
DEFAULT_CALIBRATION_RESULTS = 32
DEFAULT_CALIBRATION_BURNIN = 1
DEFAULT_CALIBRATION_REGION = (0.68, 0.72)
DEFAULT_EPSILON_REPAIR_FACTOR = 1.20
DEFAULT_MAX_EPSILON_REPAIRS = 3
DEFAULT_FINAL_SCREEN_RESULTS = 96
DEFAULT_FINAL_SCREEN_BURNIN = 8


def next_repair_epsilon(
    *,
    epsilon: float,
    acceptance_mean: float,
    lower_epsilon: float | None,
    upper_epsilon: float | None,
    calibration_region: tuple[float, float] = DEFAULT_CALIBRATION_REGION,
    repair_factor: float = DEFAULT_EPSILON_REPAIR_FACTOR,
) -> tuple[float, float | None, float | None, str]:
    """Return the next epsilon and updated monotone acceptance bracket."""

    step = float(epsilon)
    mean = float(acceptance_mean)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not math.isfinite(mean) or not 0.0 <= mean <= 1.0:
        raise ValueError("acceptance_mean must lie inside [0, 1]")
    region_low, region_high = (float(item) for item in calibration_region)
    factor = float(repair_factor)
    if not 0.0 < region_low < region_high < 1.0:
        raise ValueError("calibration_region must be ordered inside (0, 1)")
    if not math.isfinite(factor) or factor <= 1.0:
        raise ValueError("repair_factor must exceed one")
    low = None if lower_epsilon is None else float(lower_epsilon)
    high = None if upper_epsilon is None else float(upper_epsilon)
    if mean > region_high:
        low = step if low is None else max(low, step)
        direction = "increase_epsilon"
    elif mean < region_low:
        high = step if high is None else min(high, step)
        direction = "decrease_epsilon"
    else:
        return step, low, high, "calibration_region_reached"
    if low is not None and high is not None:
        if not low < high:
            raise ValueError("epsilon repair bracket is inconsistent")
        proposal = math.sqrt(low * high)
        direction = "geometric_bracket_midpoint"
    elif direction == "increase_epsilon":
        proposal = step * factor
    else:
        proposal = step / factor
    return proposal, low, high, direction


def _state_signature(state: Any) -> str:
    serialized = bytes(
        tf.io.serialize_tensor(tf.convert_to_tensor(state, tf.float64)).numpy()
    )
    return hashlib.sha256(serialized).hexdigest()


@dataclass(frozen=True)
class NeuTraStateContinuingBroadGridConfig:
    """Execution controls for the state-continuing epsilon-repair route."""

    initial_step_size: float
    root_seed: tuple[int, int]
    evidence_path: str
    initial_epsilon_by_l: Mapping[int, float] | None = None
    adaptation_steps: int = DEFAULT_ADAPTATION_STEPS
    post_adaptation_results: int = DEFAULT_POST_ADAPTATION_RESULTS
    calibration_results: int = DEFAULT_CALIBRATION_RESULTS
    calibration_burnin: int = DEFAULT_CALIBRATION_BURNIN
    calibration_region: tuple[float, float] = DEFAULT_CALIBRATION_REGION
    epsilon_repair_factor: float = DEFAULT_EPSILON_REPAIR_FACTOR
    max_epsilon_repairs: int = DEFAULT_MAX_EPSILON_REPAIRS
    final_screen_results: int = DEFAULT_FINAL_SCREEN_RESULTS
    final_screen_burnin: int = DEFAULT_FINAL_SCREEN_BURNIN
    use_xla: bool = True
    required_status_keys: tuple[str, ...] = DEFAULT_REQUIRED_STATUS_KEYS

    def __post_init__(self) -> None:
        step = float(self.initial_step_size)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("initial_step_size must be positive and finite")
        seed = tuple(int(item) for item in self.root_seed)
        if len(seed) != 2 or any(item < 0 for item in seed):
            raise ValueError("root_seed must contain two nonnegative integers")
        evidence_path = str(self.evidence_path)
        if not evidence_path:
            raise ValueError("evidence_path must be non-empty")
        warm = self.initial_epsilon_by_l
        if warm is not None:
            normalized_warm = {}
            for key, value in dict(warm).items():
                leapfrog = int(key)
                epsilon = float(value)
                if leapfrog <= 0 or not math.isfinite(epsilon) or epsilon <= 0.0:
                    raise ValueError("initial_epsilon_by_l entries must be positive")
                normalized_warm[leapfrog] = epsilon
            object.__setattr__(self, "initial_epsilon_by_l", normalized_warm)
        for name in (
            "adaptation_steps",
            "post_adaptation_results",
            "calibration_results",
            "calibration_burnin",
            "max_epsilon_repairs",
            "final_screen_results",
            "final_screen_burnin",
        ):
            value = int(getattr(self, name))
            if value <= 0 and name != "max_epsilon_repairs":
                raise ValueError(f"{name} must be positive")
            if name == "max_epsilon_repairs" and value < 0:
                raise ValueError("max_epsilon_repairs must be nonnegative")
            object.__setattr__(self, name, value)
        if int(self.final_screen_results) <= 64:
            raise ValueError(
                "final_screen_results must exceed the 64-draw nomination rung"
            )
        region = tuple(float(item) for item in self.calibration_region)
        if len(region) != 2 or not 0.0 < region[0] < region[1] < 1.0:
            raise ValueError("calibration_region must be ordered inside (0, 1)")
        factor = float(self.epsilon_repair_factor)
        if not math.isfinite(factor) or factor <= 1.0:
            raise ValueError("epsilon_repair_factor must exceed one")
        object.__setattr__(self, "initial_step_size", step)
        object.__setattr__(self, "root_seed", seed)
        object.__setattr__(self, "evidence_path", evidence_path)
        object.__setattr__(self, "calibration_region", region)
        object.__setattr__(self, "epsilon_repair_factor", factor)
        object.__setattr__(self, "use_xla", bool(self.use_xla))
        object.__setattr__(
            self,
            "required_status_keys",
            _normalized_status_keys(self.required_status_keys),
        )

    def warm_start_epsilon(self, leapfrog: int) -> float:
        warm = self.initial_epsilon_by_l
        if warm is not None and int(leapfrog) in warm:
            return float(warm[int(leapfrog)])
        return float(self.initial_step_size)

    def payload(self) -> Mapping[str, Any]:
        return {
            "initial_step_size": self.initial_step_size,
            "initial_epsilon_by_l": (
                None
                if self.initial_epsilon_by_l is None
                else {str(key): value for key, value in self.initial_epsilon_by_l.items()}
            ),
            "initial_epsilon_role": "warm_start_hypothesis_only",
            "root_seed": self.root_seed,
            "adaptation_steps": self.adaptation_steps,
            "post_adaptation_results": self.post_adaptation_results,
            "calibration_results": self.calibration_results,
            "calibration_burnin": self.calibration_burnin,
            "calibration_region": self.calibration_region,
            "epsilon_repair_factor": self.epsilon_repair_factor,
            "max_epsilon_repairs": self.max_epsilon_repairs,
            "final_screen_results": self.final_screen_results,
            "final_screen_burnin": self.final_screen_burnin,
            "use_xla": self.use_xla,
            "required_status_keys": self.required_status_keys,
            "evidence_path": self.evidence_path,
            "primary_l_grid": PRIMARY_L_GRID,
            "chain_count": CHAIN_COUNT,
            "replication_count": REPLICATION_COUNT,
        }


class NeuTraStateContinuingBroadGridCallbacks:
    """Per-``L`` adaptation, calibration repair, and disjoint final screens."""

    def __init__(
        self,
        *,
        adapter: Any,
        policy: OperationalBroadGridPolicy,
        handoff: OperationalMassHandoff,
        config: NeuTraStateContinuingBroadGridConfig,
    ) -> None:
        self.adapter = adapter
        self.policy = policy
        self.handoff = handoff
        self.config = config
        offsets = tf.constant((0.0, 0.1, -0.1, 0.16), tf.float64)[:, None]
        self.initial_state = tf.broadcast_to(
            offsets, (CHAIN_COUNT, int(adapter.parameter_dim))
        )
        scope = str(getattr(adapter, "target_scope", ""))
        if not scope:
            raise ValueError("state-continuing adapter must expose target_scope")
        self.events: list[Mapping[str, Any]] = []
        self._calibrated_states: dict[str, Any] = {}
        self._adapt_runner = build_reusable_full_chain_tfp_hmc_runner(
            adapter,
            self.initial_state,
            FullChainHMCConfig(
                num_results=config.post_adaptation_results,
                num_burnin_steps=config.adaptation_steps,
                step_size=config.initial_step_size,
                num_leapfrog_steps=PRIMARY_L_GRID[0],
                seed=policy.root_seed,
                use_xla=config.use_xla,
                trace_policy="standard",
                target_status_trace_policy="none",
                tuning_policy=HMCTuningPolicy.fixed_mass_dual_averaging(
                    num_adaptation_steps=config.adaptation_steps,
                    target_accept_prob=0.70,
                    source=config.evidence_path,
                ),
                target_scope=scope,
            ),
            dynamic_num_leapfrog_steps=True,
        )
        self._calibration_runner = build_reusable_full_chain_tfp_hmc_runner(
            adapter,
            self.initial_state,
            FullChainHMCConfig(
                num_results=config.calibration_results,
                num_burnin_steps=config.calibration_burnin,
                step_size=config.initial_step_size,
                num_leapfrog_steps=PRIMARY_L_GRID[0],
                seed=policy.root_seed,
                use_xla=config.use_xla,
                trace_policy="standard",
                target_status_trace_policy="none",
                target_scope=scope,
            ),
            dynamic_num_leapfrog_steps=True,
        )
        self._final_screen_runner = build_reusable_full_chain_tfp_hmc_runner(
            adapter,
            self.initial_state,
            FullChainHMCConfig(
                num_results=config.final_screen_results,
                num_burnin_steps=config.final_screen_burnin,
                step_size=config.initial_step_size,
                num_leapfrog_steps=PRIMARY_L_GRID[0],
                seed=policy.root_seed,
                use_xla=config.use_xla,
                trace_policy="standard",
                target_status_trace_policy="none",
                target_scope=scope,
            ),
            dynamic_num_leapfrog_steps=True,
        )

    def _summarize_run(
        self,
        *,
        result: Any,
        role: str,
        leapfrog: int,
        epsilon: float,
        seed: tuple[int, int],
        wall_seconds: float,
        initial_state_signature: str,
    ) -> Mapping[str, Any]:
        evaluation = evaluate_fixed_screen_run(
            self.adapter,
            result,
            chain_count=CHAIN_COUNT,
            required_status_keys=self.config.required_status_keys,
        )
        chain_means = evaluation["chain_means"]
        final_state = tf.convert_to_tensor(result.samples[-1], tf.float64)
        row = {
            "role": role,
            "num_leapfrog_steps": int(leapfrog),
            "step_size": float(epsilon),
            "seed": seed,
            "chain_means": chain_means,
            "grand_mean": math.fsum(chain_means) / len(chain_means),
            "hard_rejection_reasons": evaluation["hard_rejection_reasons"],
            "native_divergence_status": evaluation["native_divergence_status"],
            "native_divergence_count": evaluation["native_divergence_count"],
            "telemetry": _json_ready(evaluation["telemetry"]),
            "wall_seconds": float(wall_seconds),
            "runner_metadata": result.metadata,
            "initial_state_signature": initial_state_signature,
            "final_state_signature": _state_signature(final_state),
            "all_draws_discarded": True,
        }
        self.events.append(row)
        return {**row, "final_state": final_state}

    def _fixed_run(
        self,
        *,
        runner: Any,
        current_state: Any,
        leapfrog: int,
        epsilon: float,
        seed: tuple[int, int],
        role: str,
    ) -> Mapping[str, Any]:
        initial_signature = _state_signature(current_state)
        started = time.perf_counter()
        result = runner.run(
            current_state=current_state,
            seed=seed,
            step_size=float(epsilon),
            num_leapfrog_steps=int(leapfrog),
        )
        return self._summarize_run(
            result=result,
            role=role,
            leapfrog=leapfrog,
            epsilon=epsilon,
            seed=seed,
            wall_seconds=time.perf_counter() - started,
            initial_state_signature=initial_signature,
        )

    def calibrate_primary(self, request: Any) -> tuple[float, Any, Mapping[str, Any]]:
        if request.mass_handoff_signature != self.handoff.signature:
            raise ValueError("primary request mass handoff changed")
        leapfrog = int(request.num_leapfrog_steps)
        initial_epsilon = self.config.warm_start_epsilon(leapfrog)
        initial_signature = _state_signature(self.initial_state)
        started = time.perf_counter()
        adapted = self._adapt_runner.run(
            current_state=self.initial_state,
            seed=request.tune_seed,
            step_size=initial_epsilon,
            num_leapfrog_steps=leapfrog,
        )
        step_trace = tf.reshape(
            tf.convert_to_tensor(adapted.trace.get("step_size"), tf.float64), [-1]
        )
        if int(step_trace.shape[0]) != self.config.post_adaptation_results:
            raise ValueError("post-adaptation step trace has the wrong length")
        if not bool(tf.reduce_all(tf.math.is_finite(step_trace)).numpy()):
            raise ValueError("dual averaging produced nonfinite epsilon")
        epsilon = float(step_trace[-1].numpy())
        if not bool(
            tf.reduce_all(
                tf.equal(step_trace, tf.fill(tf.shape(step_trace), step_trace[-1]))
            ).numpy()
        ):
            raise ValueError("post-adaptation draws did not use one frozen epsilon")
        adaptation_row = self._summarize_run(
            result=adapted,
            role="state_continuing_adaptation_calibration",
            leapfrog=leapfrog,
            epsilon=epsilon,
            seed=request.tune_seed,
            wall_seconds=time.perf_counter() - started,
            initial_state_signature=initial_signature,
        )
        if adaptation_row["hard_rejection_reasons"]:
            raise ValueError("adaptation calibration failed a hard health gate")
        current_state = adaptation_row["final_state"]
        calibration_rows = [
            {key: value for key, value in adaptation_row.items() if key != "final_state"}
        ]
        acceptance_mean = float(adaptation_row["grand_mean"])
        region = self.config.calibration_region
        lower_epsilon = None
        upper_epsilon = None
        for repair_index in range(self.config.max_epsilon_repairs):
            if region[0] <= acceptance_mean <= region[1]:
                break
            next_epsilon, lower_epsilon, upper_epsilon, action = next_repair_epsilon(
                epsilon=epsilon,
                acceptance_mean=acceptance_mean,
                lower_epsilon=lower_epsilon,
                upper_epsilon=upper_epsilon,
                calibration_region=region,
                repair_factor=self.config.epsilon_repair_factor,
            )
            repair_seed = operational_broad_seed(
                self.policy.root_seed,
                domain="state_continuing_epsilon_repair_calibration",
                num_leapfrog_steps=leapfrog,
                epsilon=next_epsilon,
                replication_index=repair_index,
            )
            repair_row = self._fixed_run(
                runner=self._calibration_runner,
                current_state=current_state,
                leapfrog=leapfrog,
                epsilon=next_epsilon,
                seed=repair_seed,
                role="state_continuing_epsilon_repair_calibration",
            )
            if repair_row["hard_rejection_reasons"]:
                raise ValueError("epsilon repair calibration failed a hard health gate")
            repair_public = {
                key: value for key, value in repair_row.items() if key != "final_state"
            }
            repair_public["repair_index"] = repair_index
            repair_public["repair_action"] = action
            repair_public["lower_epsilon_bound"] = lower_epsilon
            repair_public["upper_epsilon_bound"] = upper_epsilon
            calibration_rows.append(repair_public)
            epsilon = next_epsilon
            acceptance_mean = float(repair_row["grand_mean"])
            current_state = repair_row["final_state"]
        calibrated_state_signature = _state_signature(current_state)
        tune_event = {
            "role": "state_continuing_epsilon_tune",
            "num_leapfrog_steps": leapfrog,
            "initial_step_size": initial_epsilon,
            "tuned_step_size": epsilon,
            "target_accept_prob": 0.70,
            "calibration_region": region,
            "calibration_final_mean": acceptance_mean,
            "calibration_region_reached": region[0] <= acceptance_mean <= region[1],
            "calibration_rows": tuple(calibration_rows),
            "calibrated_state_signature": calibrated_state_signature,
            "state_continuation_performed": True,
            "all_draws_discarded": True,
        }
        self.events.append(tune_event)
        return epsilon, current_state, tune_event

    def primary(self, request: Any) -> OperationalPrimaryCandidate:
        leapfrog = int(request.num_leapfrog_steps)
        epsilon, current_state, tune_event = self.calibrate_primary(request)
        screens = tuple(
            self._fixed_run(
                runner=self._final_screen_runner,
                current_state=current_state,
                leapfrog=leapfrog,
                epsilon=epsilon,
                seed=operational_broad_seed(
                    self.policy.root_seed,
                    domain="state_continuing_primary_fresh_screen",
                    num_leapfrog_steps=leapfrog,
                    epsilon=epsilon,
                    replication_index=index,
                ),
                role="state_continuing_primary_fresh_screen",
            )
            for index in range(REPLICATION_COUNT)
        )
        screen_payloads = tuple(
            {key: value for key, value in row.items() if key != "final_state"}
            for row in screens
        )
        reasons = tuple(
            dict.fromkeys(
                reason
                for screen in screen_payloads
                for reason in screen["hard_rejection_reasons"]
            )
        )
        evidence = classify_operational_pair_evidence(
            chain_run_means=tuple(
                value for screen in screen_payloads for value in screen["chain_means"]
            ),
            evidence_signature=_signature(
                "bayesfilter.neutra_state_continuing_broad_grid.primary_evidence.v1",
                {"tune": tune_event, "screens": screen_payloads},
            ),
            policy=self.policy,
            hard_rejection_reasons=reasons,
        )
        candidate = OperationalPrimaryCandidate(
            request=request,
            tuned_step_size=epsilon,
            evidence=evidence,
            metric_signature=self.handoff.frozen_metric_signature,
            coordinate_signature=self.handoff.coordinate_signature,
            lineage_signature=self.handoff.lineage_signature,
            tune_evidence_signature=_signature(
                "bayesfilter.neutra_state_continuing_broad_grid.epsilon_tune.v1",
                tune_event,
            ),
        )
        self._calibrated_states[candidate.signature] = tf.identity(current_state)
        return candidate

    def guard(self, request: Any) -> SameEpsilonNeighborGuard:
        if request.mass_handoff_signature != self.handoff.signature:
            raise ValueError("coverage request mass handoff changed")
        parent_signatures = tuple(sorted(request.parent_candidate_signatures))
        if len(parent_signatures) != 1 or any(
            signature not in self._calibrated_states for signature in parent_signatures
        ):
            raise ValueError("coverage probe requires exactly one calibrated parent state")
        current_state = self._calibrated_states[parent_signatures[0]]
        screens = tuple(
            self._fixed_run(
                runner=self._final_screen_runner,
                current_state=current_state,
                leapfrog=int(request.num_leapfrog_steps),
                epsilon=float(request.inherited_step_size),
                seed=seed,
                role="state_continuing_same_epsilon_neighbor_guard",
            )
            for seed in request.screen_seeds
        )
        screen_payloads = tuple(
            {key: value for key, value in row.items() if key != "final_state"}
            for row in screens
        )
        reasons = tuple(
            dict.fromkeys(
                reason
                for screen in screen_payloads
                for reason in screen["hard_rejection_reasons"]
            )
        )
        evidence = classify_operational_pair_evidence(
            chain_run_means=tuple(
                value for screen in screen_payloads for value in screen["chain_means"]
            ),
            evidence_signature=_signature(
                "bayesfilter.neutra_state_continuing_broad_grid.coverage_evidence.v1",
                {"request": request.payload(), "screens": screen_payloads},
            ),
            policy=self.policy,
            hard_rejection_reasons=reasons,
        )
        return SameEpsilonNeighborGuard(request=request, evidence=evidence)


def run_neutra_state_continuing_broad_grid_tuning(
    *,
    adapter: Any,
    target_signature: str,
    config: NeuTraStateContinuingBroadGridConfig,
    output_dir: Path,
    callbacks_factory: Callable[..., Any] = NeuTraStateContinuingBroadGridCallbacks,
    procedure_metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Run the state-continuing repair barriers and emit private/public artifacts."""

    if procedure_metadata is None:
        from bayesfilter.inference.neutra_shared_procedure import (
            state_continuing_procedure_metadata,
        )

        procedure_metadata = state_continuing_procedure_metadata()
    metadata = dict(procedure_metadata)
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"broad-grid output directory must be fresh: {root}")
    root.mkdir(parents=True)
    handoff = build_fixed_identity_broad_grid_handoff(
        adapter=adapter,
        target_signature=target_signature,
        evidence_path=config.evidence_path,
    )
    policy = OperationalBroadGridPolicy(
        root_seed=config.root_seed,
        confirmation_num_results=config.final_screen_results,
        chain_count=CHAIN_COUNT,
        replication_count=REPLICATION_COUNT,
    )
    callbacks = callbacks_factory(
        adapter=adapter,
        policy=policy,
        handoff=handoff,
        config=config,
    )
    result = run_operational_broad_grid(
        policy=policy,
        handoff=handoff,
        primary_runner=callbacks.primary,
        guard_runner=callbacks.guard,
    )
    private_payload = {
        **result.payload(),
        **metadata,
        "execution_config": config.payload(),
        "events": tuple(callbacks.events),
    }
    public_payload = {
        **result.public_payload(),
        **metadata,
        "execution_config": {
            key: value
            for key, value in config.payload().items()
            if key not in ("initial_step_size", "initial_epsilon_by_l")
        },
    }
    atomic_write_json(root / "private_result.json", private_payload)
    atomic_write_json(root / "public_result.json", public_payload)
    return {
        "private": private_payload,
        "public": public_payload,
        "private_result_path": str(root / "private_result.json"),
        "public_result_path": str(root / "public_result.json"),
    }
