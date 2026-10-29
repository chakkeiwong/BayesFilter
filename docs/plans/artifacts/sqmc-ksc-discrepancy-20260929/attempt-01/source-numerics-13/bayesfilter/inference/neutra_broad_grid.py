"""Generic broad-grid HMC tuning for one frozen NeuTra transport.

The application callbacks delegate every transition to the shared TensorFlow /
TensorFlow Probability HMC runner. This module only binds that runner to the
reviewed broad-grid orchestration and writes tuning-only evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import tensorflow as tf

from bayesfilter.inference.hmc import (
    FullChainHMCConfig,
    build_reusable_full_chain_tfp_hmc_runner,
    stable_adapter_signature,
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
from bayesfilter.inference.hmc_verification import summarize_hmc_tuning_telemetry
from bayesfilter.runtime import atomic_write_json


DEFAULT_TUNE_BURNIN = 64
DEFAULT_TUNE_RESULTS = 1
DEFAULT_SCREEN_BURNIN = 1
DEFAULT_SCREEN_RESULTS = 128
CHAIN_COUNT = 4
REPLICATION_COUNT = 3

from bayesfilter.inference.neutra_shared_procedure import (  # noqa: E402
    BASE_REQUIRED_STATUS_KEYS,
    DEFAULT_REQUIRED_STATUS_KEYS,
    _normalized_status_keys,
)


def combined_target_health(
    adapter: Any,
    samples: Any,
    *,
    required_status_keys: tuple[str, ...] = DEFAULT_REQUIRED_STATUS_KEYS,
) -> tuple[str, ...]:
    """Hard target-validity reasons for one fixed-kernel run's samples.

    Presence is enforced for every declared required key; the extended UKF
    numeric telemetry checks apply to whichever of those keys the target
    actually returns, so a target whose contract lacks innovation telemetry is
    not misclassified as unhealthy while a target that provides it is always
    checked.
    """

    keys = _normalized_status_keys(required_status_keys)
    tensor = tf.convert_to_tensor(samples, tf.float64)
    flat = tf.reshape(tensor, (-1, int(adapter.parameter_dim)))
    value, score = adapter.log_prob_and_grad(flat)
    status = adapter.target_status_telemetry(flat)
    reasons: list[str] = []
    if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
        reasons.append("nonfinite_candidate_state")
    if not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
        reasons.append("nonfinite_target_log_prob")
    if not bool(tf.reduce_all(tf.math.is_finite(score)).numpy()):
        reasons.append("nonfinite_target_score")
    if any(name not in status for name in keys):
        return tuple((*reasons, "target_status_telemetry_failure"))
    status_ok = tf.reduce_all(
        tf.logical_and(
            tf.equal(tf.convert_to_tensor(status["status_code"]), 0),
            tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool),
        )
    )
    numeric_ok = tf.constant(True)
    if "floor_count_value" in status:
        numeric_ok = tf.logical_and(
            numeric_ok,
            tf.reduce_all(tf.convert_to_tensor(status["floor_count_value"]) >= 0),
        )
    for name in ("min_innovation_eigenvalue", "innovation_condition_estimate"):
        if name in status:
            numeric_ok = tf.logical_and(
                numeric_ok,
                tf.reduce_all(
                    tf.math.is_finite(tf.convert_to_tensor(status[name], tf.float64))
                ),
            )
    if not bool(tf.logical_and(status_ok, numeric_ok).numpy()):
        reasons.append("target_status_telemetry_failure")
    return tuple(dict.fromkeys(reasons))


def evaluate_fixed_screen_run(
    adapter: Any,
    result: Any,
    *,
    chain_count: int,
    required_status_keys: tuple[str, ...] = DEFAULT_REQUIRED_STATUS_KEYS,
) -> Mapping[str, Any]:
    """Shared hard gates and acceptance chain means for one fixed screen run."""

    reasons = list(
        combined_target_health(
            adapter, result.samples, required_status_keys=required_status_keys
        )
    )
    trace = result.trace
    required = ("log_accept_ratio", "is_accepted", "target_log_prob")
    if any(name not in trace for name in required):
        raise ValueError("required_standard_trace_missing")
    telemetry = summarize_hmc_tuning_telemetry(
        samples=result.samples,
        log_accept_ratio=trace["log_accept_ratio"],
        is_accepted=trace["is_accepted"],
    )
    chain_means_tensor = tf.reshape(
        telemetry["mean_acceptance_probability_by_chain"], [-1]
    )
    chain_means = tuple(float(item) for item in chain_means_tensor.numpy().tolist())
    if len(chain_means) != int(chain_count) or any(
        not math.isfinite(item) for item in chain_means
    ):
        reasons.append("nonfinite_log_accept_ratio")
    for name in ("log_accept_ratio", "target_log_prob"):
        if not bool(tf.reduce_all(tf.math.is_finite(trace[name])).numpy()):
            reasons.append(f"nonfinite_{name}")
    movement = tf.convert_to_tensor(telemetry["movement_rate_by_chain"], tf.float64)
    repeated = tf.convert_to_tensor(
        telemetry["repeated_state_fraction_by_chain"], tf.float64
    )
    normalized = tf.convert_to_tensor(
        telemetry["normalized_return_displacement_by_chain"], tf.float64
    )
    path_return = tf.convert_to_tensor(
        telemetry["path_return_fraction_by_chain"], tf.float64
    )
    movement_ok = tf.reduce_all(
        tf.logical_and(
            tf.logical_and(movement >= 0.05, repeated <= 0.95),
            normalized >= 1.0e-4,
        )
    )
    if not bool(movement_ok.numpy()):
        reasons.append("movement_gate_failed")
    if not bool(tf.reduce_all(path_return <= 0.95).numpy()):
        reasons.append("path_return_resonance_detected")
    divergence_status = str(
        result.diagnostics.get("native_divergence_status", "not_exposed_by_kernel")
    )
    divergence_count = result.diagnostics.get("divergence_count")
    if hasattr(divergence_count, "numpy"):
        divergence_count = int(divergence_count.numpy())
    if divergence_status == "available" and int(divergence_count or 0) > 0:
        reasons.append("native_divergence_positive")
    return {
        "hard_rejection_reasons": tuple(dict.fromkeys(reasons)),
        "chain_means": chain_means,
        "telemetry": telemetry,
        "native_divergence_status": divergence_status,
        "native_divergence_count": divergence_count,
    }


@dataclass(frozen=True)
class NeuTraBroadGridTuningConfig:
    """Execution controls not already frozen by the broad-grid policy."""

    initial_step_size: float
    root_seed: tuple[int, int]
    tune_burnin: int = DEFAULT_TUNE_BURNIN
    tune_results: int = DEFAULT_TUNE_RESULTS
    screen_burnin: int = DEFAULT_SCREEN_BURNIN
    screen_results: int = DEFAULT_SCREEN_RESULTS
    use_xla: bool = True
    required_status_keys: tuple[str, ...] = DEFAULT_REQUIRED_STATUS_KEYS
    evidence_path: str = (
        "docs/plans/bayesfilter-neutra-remaining-models-broad-grid-"
        "continuation-plan-2026-07-30.md"
    )

    def __post_init__(self) -> None:
        step = float(self.initial_step_size)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("initial_step_size must be positive and finite")
        seed = tuple(int(item) for item in self.root_seed)
        if len(seed) != 2 or any(item < 0 for item in seed):
            raise ValueError("root_seed must contain two nonnegative integers")
        for name in ("tune_burnin", "tune_results", "screen_burnin", "screen_results"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if int(self.screen_results) <= 64:
            raise ValueError("screen_results must exceed the 64-draw nomination rung")
        evidence_path = str(self.evidence_path)
        if not evidence_path:
            raise ValueError("evidence_path must be non-empty")
        object.__setattr__(self, "initial_step_size", step)
        object.__setattr__(self, "root_seed", seed)
        object.__setattr__(self, "use_xla", bool(self.use_xla))
        object.__setattr__(
            self,
            "required_status_keys",
            _normalized_status_keys(self.required_status_keys),
        )
        object.__setattr__(self, "evidence_path", evidence_path)

    def payload(self) -> Mapping[str, Any]:
        return {
            "initial_step_size": self.initial_step_size,
            "root_seed": self.root_seed,
            "tune_burnin": self.tune_burnin,
            "tune_results": self.tune_results,
            "screen_burnin": self.screen_burnin,
            "screen_results": self.screen_results,
            "use_xla": self.use_xla,
            "required_status_keys": self.required_status_keys,
            "evidence_path": self.evidence_path,
            "primary_l_grid": PRIMARY_L_GRID,
            "chain_count": CHAIN_COUNT,
            "replication_count": REPLICATION_COUNT,
        }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy().tolist())
    return value


def _signature(label: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"label": label, "payload": _json_ready(payload)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def build_fixed_identity_broad_grid_handoff(
    *,
    adapter: Any,
    target_signature: str,
    evidence_path: str,
) -> OperationalMassHandoff:
    """Issue fixed-identity metric and coordinate lineage for one transport."""

    adapter_signature = stable_adapter_signature(adapter)
    transport_hash = str(getattr(adapter, "transport_manifest_hash", ""))
    if not transport_hash:
        raise ValueError("fixed-transport adapter must expose transport_manifest_hash")
    target = str(target_signature)
    if not target:
        raise ValueError("target_signature must be non-empty")
    identity_metric = _signature(
        "bayesfilter.neutra_broad_grid.fixed_identity_metric.v1",
        {"dimension": int(adapter.parameter_dim), "coordinate": transport_hash},
    )
    coordinate = _signature(
        "bayesfilter.neutra_broad_grid.frozen_transport_coordinate.v1",
        {"transport_manifest_hash": transport_hash},
    )
    lineage = _signature(
        "bayesfilter.neutra_broad_grid.lineage.v1",
        {
            "target_signature": target,
            "adapter_signature": adapter_signature,
            "transport_manifest_hash": transport_hash,
            "coordinate_signature": coordinate,
            "metric_signature": identity_metric,
            "chain_offsets": (0.0, 0.1, -0.1, 0.16),
            "dtype": "float64",
        },
    )
    return OperationalMassHandoff(
        update_disposition="fixed_identity",
        prior_metric_signature=identity_metric,
        frozen_metric_signature=identity_metric,
        coordinate_signature=coordinate,
        adapter_signature=adapter_signature,
        target_signature=target,
        lineage_signature=lineage,
        canonical_covariance_signature=identity_metric,
        latent_metric_signature=identity_metric,
        metric_evidence_signature=_signature(
            "bayesfilter.neutra_broad_grid.fixed_identity_policy_evidence.v1",
            {"evidence_path": str(evidence_path), "mass_updates": False},
        ),
        retained_prior_metric=True,
        latent_identity_equivalence_proven=True,
    )


class NeuTraBroadGridCallbacks:
    """Target-agnostic dual-averaging and fixed-screen callbacks."""

    def __init__(
        self,
        *,
        adapter: Any,
        policy: OperationalBroadGridPolicy,
        handoff: OperationalMassHandoff,
        config: NeuTraBroadGridTuningConfig,
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
            raise ValueError("broad-grid adapter must expose target_scope")
        self.events: list[Mapping[str, Any]] = []
        self._tune_runner = build_reusable_full_chain_tfp_hmc_runner(
            adapter,
            self.initial_state,
            FullChainHMCConfig(
                num_results=config.tune_results,
                num_burnin_steps=config.tune_burnin,
                step_size=config.initial_step_size,
                num_leapfrog_steps=PRIMARY_L_GRID[0],
                seed=policy.root_seed,
                use_xla=config.use_xla,
                trace_policy="standard",
                target_status_trace_policy="none",
                tuning_policy=HMCTuningPolicy.fixed_mass_dual_averaging(
                    num_adaptation_steps=config.tune_burnin,
                    target_accept_prob=0.70,
                    source=config.evidence_path,
                ),
                target_scope=scope,
            ),
            dynamic_num_leapfrog_steps=True,
        )
        self._screen_runner = build_reusable_full_chain_tfp_hmc_runner(
            adapter,
            self.initial_state,
            FullChainHMCConfig(
                num_results=config.screen_results,
                num_burnin_steps=config.screen_burnin,
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

    def _combined_health(self, samples: Any) -> tuple[str, ...]:
        return combined_target_health(
            self.adapter,
            samples,
            required_status_keys=self.config.required_status_keys,
        )

    def _screen(
        self,
        *,
        leapfrog: int,
        epsilon: float,
        seed: tuple[int, int],
        role: str,
    ) -> Mapping[str, Any]:
        started = time.perf_counter()
        result = self._screen_runner.run(
            seed=seed,
            step_size=float(epsilon),
            num_leapfrog_steps=int(leapfrog),
        )
        evaluation = evaluate_fixed_screen_run(
            self.adapter,
            result,
            chain_count=CHAIN_COUNT,
            required_status_keys=self.config.required_status_keys,
        )
        row = {
            "role": role,
            "num_leapfrog_steps": int(leapfrog),
            "step_size": float(epsilon),
            "seed": seed,
            "chain_means": evaluation["chain_means"],
            "hard_rejection_reasons": evaluation["hard_rejection_reasons"],
            "native_divergence_status": evaluation["native_divergence_status"],
            "native_divergence_count": evaluation["native_divergence_count"],
            "telemetry": _json_ready(evaluation["telemetry"]),
            "wall_seconds": time.perf_counter() - started,
            "runner_metadata": result.metadata,
            "all_draws_discarded": True,
        }
        self.events.append(row)
        return row

    def primary(self, request: Any) -> OperationalPrimaryCandidate:
        if request.mass_handoff_signature != self.handoff.signature:
            raise ValueError("primary request mass handoff changed")
        started = time.perf_counter()
        tune = self._tune_runner.run(
            seed=request.tune_seed,
            step_size=self.config.initial_step_size,
            num_leapfrog_steps=request.num_leapfrog_steps,
        )
        step_trace = tf.convert_to_tensor(tune.trace.get("step_size"), tf.float64)
        if step_trace.shape.rank not in {1, 2}:
            raise ValueError("dual averaging did not expose a scalar step trace")
        final_step = tf.reshape(step_trace[-1], [-1])
        if not bool(tf.reduce_all(tf.math.is_finite(final_step)).numpy()):
            raise ValueError("dual averaging produced nonfinite epsilon")
        spread = float((tf.reduce_max(final_step) - tf.reduce_min(final_step)).numpy())
        scale = max(1.0, float(tf.reduce_max(tf.abs(final_step)).numpy()))
        if spread > 1.0e-10 * scale:
            raise ValueError("dual averaging did not produce one common epsilon")
        epsilon = float(tf.reduce_mean(final_step).numpy())
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("dual averaging produced invalid epsilon")
        tune_health = self._combined_health(tune.samples)
        if tune_health:
            raise ValueError(
                "dual-averaging target health failed: " + ",".join(tune_health)
            )
        tune_event = {
            "role": "independent_epsilon_tune",
            "num_leapfrog_steps": request.num_leapfrog_steps,
            "initial_step_size": self.config.initial_step_size,
            "tuned_step_size": epsilon,
            "seed": request.tune_seed,
            "wall_seconds": time.perf_counter() - started,
            "runner_metadata": tune.metadata,
            "all_draws_discarded": True,
        }
        self.events.append(tune_event)
        screens = tuple(
            self._screen(
                leapfrog=request.num_leapfrog_steps,
                epsilon=epsilon,
                seed=operational_broad_seed(
                    self.policy.root_seed,
                    domain="primary_independent_fixed_screen",
                    num_leapfrog_steps=request.num_leapfrog_steps,
                    epsilon=epsilon,
                    replication_index=index,
                ),
                role="primary_independent_fixed_screen",
            )
            for index in range(REPLICATION_COUNT)
        )
        reasons = tuple(
            dict.fromkeys(
                reason
                for screen in screens
                for reason in screen["hard_rejection_reasons"]
            )
        )
        evidence_payload = {"tune": tune_event, "screens": screens}
        evidence = classify_operational_pair_evidence(
            chain_run_means=tuple(
                value for screen in screens for value in screen["chain_means"]
            ),
            evidence_signature=_signature(
                "bayesfilter.neutra_broad_grid.primary_evidence.v1",
                evidence_payload,
            ),
            policy=self.policy,
            hard_rejection_reasons=reasons,
        )
        return OperationalPrimaryCandidate(
            request=request,
            tuned_step_size=epsilon,
            evidence=evidence,
            metric_signature=self.handoff.frozen_metric_signature,
            coordinate_signature=self.handoff.coordinate_signature,
            lineage_signature=self.handoff.lineage_signature,
            tune_evidence_signature=_signature(
                "bayesfilter.neutra_broad_grid.independent_epsilon_tune.v1",
                tune_event,
            ),
        )

    def guard(self, request: Any) -> SameEpsilonNeighborGuard:
        if request.mass_handoff_signature != self.handoff.signature:
            raise ValueError("coverage request mass handoff changed")
        screens = tuple(
            self._screen(
                leapfrog=request.num_leapfrog_steps,
                epsilon=request.inherited_step_size,
                seed=seed,
                role="same_epsilon_neighbor_coverage_screen",
            )
            for seed in request.screen_seeds
        )
        reasons = tuple(
            dict.fromkeys(
                reason
                for screen in screens
                for reason in screen["hard_rejection_reasons"]
            )
        )
        evidence = classify_operational_pair_evidence(
            chain_run_means=tuple(
                value for screen in screens for value in screen["chain_means"]
            ),
            evidence_signature=_signature(
                "bayesfilter.neutra_broad_grid.same_epsilon_coverage.v1",
                {"request": request.payload(), "screens": screens},
            ),
            policy=self.policy,
            hard_rejection_reasons=reasons,
        )
        return SameEpsilonNeighborGuard(request=request, evidence=evidence)


def run_neutra_operational_broad_grid_tuning(
    *,
    adapter: Any,
    target_signature: str,
    config: NeuTraBroadGridTuningConfig,
    output_dir: Path,
    callbacks_factory: Callable[..., Any] = NeuTraBroadGridCallbacks,
    procedure_metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Run the complete tuning barriers and emit private/public artifacts."""

    if procedure_metadata is None:
        from bayesfilter.inference.neutra_shared_procedure import (
            operational_procedure_metadata,
        )

        procedure_metadata = operational_procedure_metadata()
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
        confirmation_num_results=config.screen_results,
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
            if key != "initial_step_size"
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
