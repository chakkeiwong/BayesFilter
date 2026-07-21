#!/usr/bin/env python3
"""Run the reviewed PP-UKF fixed-identity broad L/epsilon tuning campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = "docs/plans/bayesfilter-pp-ukf-operational-broad-grid-tuning-plan-2026-07-21.md"
WARM_START_SOURCE = (
    "docs/plans/artifacts/bayesfilter-pp-ukf-offline-tuning-only-20260721-01/"
    "PP-UKF/result.json"
)
TARGET_SIGNATURE = "d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5"
TRANSPORT_SHA256 = "b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221"
PRIMARY_L_GRID = (3, 5, 9, 13, 18, 25)
INITIAL_STEP_SIZE = 0.9853849721883557
TUNE_BURNIN = 64
TUNE_RESULTS = 1
SCREEN_BURNIN = 1
SCREEN_RESULTS = 128
REPLICATION_COUNT = 3
CHAIN_COUNT = 4
CAMPAIGN_CAP_SECONDS = 4.0 * 3600.0
PRIOR_CHARGED_SECONDS = 2820.0 + 842.5780625240004 + 267.797653088026
PROJECTION_MARGIN = 1.50


class PPUKFBatchNativeBoundAdapter:
    """Expose the repository-bound PP-UKF batch target to HMC consumers."""

    def __init__(self, base_adapter: Any, *, target_signature: str) -> None:
        from bayesfilter.inference.neutra_batching import (
            bind_batch_native_neutra_target,
        )

        self.base_adapter = base_adapter
        self.binding = bind_batch_native_neutra_target(
            base_adapter, target_signature=target_signature
        )
        self.parameter_dim = int(base_adapter.parameter_dim)
        names = getattr(base_adapter, "parameter_names", ())
        self.parameter_names = tuple(names() if callable(names) else names)
        self.target_signature = str(target_signature)
        self.supports_retained_flat_batch = True
        self.supports_retained_value_score_status = True

    def adapter_signature(self) -> str:
        return self.binding.adapter_signature

    def value_score_capability(self) -> Any:
        from bayesfilter.inference.posterior_adapter import value_score_capability

        return value_score_capability(self.base_adapter)

    def log_prob_and_grad(self, theta: Any) -> tuple[Any, Any]:
        value, score, _status = self.log_prob_and_grad_status(theta)
        return value, score

    def log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[Any, Any, Mapping[str, Any]]:
        import tensorflow as tf

        values = tf.convert_to_tensor(theta, tf.float64)
        if values.shape.rank == 1:
            value, score, status = self.binding.invoke(values[tf.newaxis, :])
            return (
                tf.convert_to_tensor(value, tf.float64)[0],
                tf.convert_to_tensor(score, tf.float64)[0],
                {
                    str(name): tf.convert_to_tensor(item)[0]
                    for name, item in status.items()
                },
            )
        if values.shape.rank != 2:
            raise ValueError("PP-UKF target requires rank-1 or rank-2 positions")
        value, score, status = self.binding.invoke(values)
        return value, score, dict(status)


def build_pp_ukf_bound_adapter() -> PPUKFBatchNativeBoundAdapter:
    """Construct the frozen PP-UKF target without importing all-model campaigns."""

    from bayesfilter.ssm import stable_ssm_target_signature
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        generate_frozen_predator_prey_dataset_tf,
        make_predator_prey_ukf_neutra_adapter,
    )

    _states, observations = generate_frozen_predator_prey_dataset_tf()
    base_adapter = make_predator_prey_ukf_neutra_adapter(observations=observations)
    if stable_ssm_target_signature(base_adapter.contract) != TARGET_SIGNATURE:
        raise ValueError("PP-UKF target signature drifted")
    return PPUKFBatchNativeBoundAdapter(
        base_adapter, target_signature=TARGET_SIGNATURE
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            _json_ready(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _signature(label: str, payload: Any) -> str:
    return hashlib.sha256(_canonical({"label": label, "payload": payload})).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(_json_ready(payload), indent=2, sort_keys=True).encode("ascii") + b"\n")


def _write_progress_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True).encode("ascii")
        + b"\n"
    )
    temporary.replace(path)


def primary_transition_leapfrog_work(leapfrog: int) -> int:
    return int(leapfrog) * (
        TUNE_BURNIN
        + TUNE_RESULTS
        + REPLICATION_COUNT * (SCREEN_BURNIN + SCREEN_RESULTS)
    )


def worst_case_guard_transition_leapfrog_work(
    primary_l_grid: Sequence[int] = PRIMARY_L_GRID,
) -> int:
    work = 0
    for leapfrog in primary_l_grid:
        for neighbor in (int(leapfrog) - 1, int(leapfrog) + 1):
            if 2 <= neighbor <= 25:
                work += neighbor * REPLICATION_COUNT * (SCREEN_BURNIN + SCREEN_RESULTS)
    return work


def project_remaining_campaign_seconds(
    *,
    warm_seconds_per_transition_leapfrog: float,
    canary_wall_seconds: float,
    prior_charged_seconds: float,
    margin: float = PROJECTION_MARGIN,
) -> Mapping[str, Any]:
    rate = float(warm_seconds_per_transition_leapfrog)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("warm_seconds_per_transition_leapfrog must be positive")
    remaining_primary_work = sum(
        primary_transition_leapfrog_work(leapfrog)
        for leapfrog in PRIMARY_L_GRID
        if leapfrog != PRIMARY_L_GRID[0]
    )
    projected_new = float(canary_wall_seconds) + float(margin) * rate * remaining_primary_work
    cumulative = float(prior_charged_seconds) + projected_new
    return {
        "stage": "complete_primary_barrier",
        "warm_seconds_per_transition_leapfrog": rate,
        "canary_wall_seconds": float(canary_wall_seconds),
        "remaining_primary_transition_leapfrogs": remaining_primary_work,
        "guard_transition_leapfrogs_charged": 0,
        "projection_margin": float(margin),
        "projected_new_campaign_seconds": projected_new,
        "prior_charged_seconds": float(prior_charged_seconds),
        "projected_cumulative_seconds": cumulative,
        "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
        "full_grid_authorized": cumulative <= CAMPAIGN_CAP_SECONDS,
    }


def project_guard_barrier_seconds(
    *,
    warm_seconds_per_transition_leapfrog: float,
    current_attempt_wall_seconds: float,
    prior_charged_seconds: float,
    guard_l_values: Sequence[int],
    margin: float = PROJECTION_MARGIN,
) -> Mapping[str, Any]:
    rate = float(warm_seconds_per_transition_leapfrog)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("warm_seconds_per_transition_leapfrog must be positive")
    guard_values = tuple(int(item) for item in guard_l_values)
    if any(item < 2 or item > 25 for item in guard_values):
        raise ValueError("guard L values must lie inside [2, 25]")
    work = sum(
        leapfrog * REPLICATION_COUNT * (SCREEN_BURNIN + SCREEN_RESULTS)
        for leapfrog in guard_values
    )
    projected_guard = float(margin) * rate * work
    cumulative = (
        float(prior_charged_seconds)
        + float(current_attempt_wall_seconds)
        + projected_guard
    )
    return {
        "stage": "actual_same_epsilon_neighbor_guard_barrier",
        "warm_seconds_per_transition_leapfrog": rate,
        "current_attempt_wall_seconds": float(current_attempt_wall_seconds),
        "prior_charged_seconds": float(prior_charged_seconds),
        "guard_l_values": guard_values,
        "actual_guard_count": len(guard_values),
        "guard_transition_leapfrogs": work,
        "projection_margin": float(margin),
        "projected_guard_seconds": projected_guard,
        "projected_cumulative_seconds": cumulative,
        "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
        "guard_barrier_authorized": cumulative <= CAMPAIGN_CAP_SECONDS,
    }


class PPUKFBroadGridCallbacks:
    """Application callbacks for independent epsilon tuning and fixed screens."""

    def __init__(self, *, tf: Any, adapter: Any, policy: Any, handoff: Any) -> None:
        from bayesfilter.inference.hmc import (
            FullChainHMCConfig,
            build_reusable_full_chain_tfp_hmc_runner,
        )
        from bayesfilter.inference.hmc_tuning import HMCTuningPolicy

        self.tf = tf
        self.adapter = adapter
        self.policy = policy
        self.handoff = handoff
        offsets = tf.constant((0.0, 0.1, -0.1, 0.16), tf.float64)[:, None]
        self.initial_state = tf.broadcast_to(offsets, (CHAIN_COUNT, adapter.parameter_dim))
        self.events: list[Mapping[str, Any]] = []
        self._tune_runner = build_reusable_full_chain_tfp_hmc_runner(
            adapter,
            self.initial_state,
            FullChainHMCConfig(
                num_results=TUNE_RESULTS,
                num_burnin_steps=TUNE_BURNIN,
                step_size=INITIAL_STEP_SIZE,
                num_leapfrog_steps=PRIMARY_L_GRID[0],
                seed=policy.root_seed,
                use_xla=True,
                trace_policy="standard",
                target_status_trace_policy="none",
                tuning_policy=HMCTuningPolicy.fixed_mass_dual_averaging(
                    num_adaptation_steps=TUNE_BURNIN,
                    target_accept_prob=0.70,
                    source=PLAN,
                ),
                target_scope=adapter.target_scope,
            ),
            dynamic_num_leapfrog_steps=True,
        )
        self._screen_runner = build_reusable_full_chain_tfp_hmc_runner(
            adapter,
            self.initial_state,
            FullChainHMCConfig(
                num_results=SCREEN_RESULTS,
                num_burnin_steps=SCREEN_BURNIN,
                step_size=INITIAL_STEP_SIZE,
                num_leapfrog_steps=PRIMARY_L_GRID[0],
                seed=policy.root_seed,
                use_xla=True,
                trace_policy="standard",
                target_status_trace_policy="none",
                target_scope=adapter.target_scope,
            ),
            dynamic_num_leapfrog_steps=True,
        )

    def _combined_health(self, samples: Any) -> tuple[str, ...]:
        tf = self.tf
        tensor = tf.convert_to_tensor(samples, tf.float64)
        flat = tf.reshape(tensor, (-1, self.adapter.parameter_dim))
        value, score, status = self.adapter.log_prob_and_grad_status(flat)
        reasons: list[str] = []
        if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
            reasons.append("nonfinite_candidate_state")
        if not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
            reasons.append("nonfinite_target_log_prob")
        if not bool(tf.reduce_all(tf.math.is_finite(score)).numpy()):
            reasons.append("nonfinite_target_score")
        required = (
            "status_code",
            "valid_pre_regularized_score",
            "floor_count_value",
            "min_innovation_eigenvalue",
            "innovation_condition_estimate",
        )
        if any(name not in status for name in required):
            reasons.append("target_status_telemetry_failure")
            return tuple(dict.fromkeys(reasons))
        status_code = tf.convert_to_tensor(status["status_code"])
        valid = tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool)
        floors = tf.convert_to_tensor(status["floor_count_value"])
        status_ok = tf.reduce_all(
            tf.logical_and(tf.equal(status_code, 0), valid)
        )
        floor_ok = tf.reduce_all(floors >= 0)
        condition_ok = tf.reduce_all(
            tf.math.is_finite(
                tf.convert_to_tensor(status["innovation_condition_estimate"], tf.float64)
            )
        )
        eigen_ok = tf.reduce_all(
            tf.math.is_finite(
                tf.convert_to_tensor(status["min_innovation_eigenvalue"], tf.float64)
            )
        )
        if not bool(tf.logical_and(tf.logical_and(status_ok, floor_ok), tf.logical_and(condition_ok, eigen_ok)).numpy()):
            reasons.append("target_status_telemetry_failure")
        return tuple(dict.fromkeys(reasons))

    def _screen(self, *, leapfrog: int, epsilon: float, seed: tuple[int, int], role: str) -> Mapping[str, Any]:
        from bayesfilter.inference.hmc_verification import summarize_hmc_tuning_telemetry

        tf = self.tf
        started = time.perf_counter()
        result = self._screen_runner.run(
            seed=seed,
            step_size=float(epsilon),
            num_leapfrog_steps=int(leapfrog),
        )
        reasons = list(self._combined_health(result.samples))
        trace = result.trace
        required = ("log_accept_ratio", "is_accepted", "target_log_prob")
        if any(name not in trace for name in required):
            reasons.append("required_standard_trace_missing")
            chain_means = (float("nan"),) * CHAIN_COUNT
            telemetry_payload: Mapping[str, Any] = {}
        else:
            telemetry = summarize_hmc_tuning_telemetry(
                samples=result.samples,
                log_accept_ratio=trace["log_accept_ratio"],
                is_accepted=trace["is_accepted"],
            )
            telemetry_payload = _json_ready(telemetry)
            chain_means = tuple(
                float(item)
                for item in tf.reshape(
                    telemetry["mean_acceptance_probability_by_chain"], [-1]
                ).numpy().tolist()
            )
            if len(chain_means) != CHAIN_COUNT or any(
                not math.isfinite(item) for item in chain_means
            ):
                reasons.append("nonfinite_log_accept_ratio")
            for name in ("log_accept_ratio", "target_log_prob"):
                if not bool(tf.reduce_all(tf.math.is_finite(trace[name])).numpy()):
                    reasons.append(f"nonfinite_{name}")
            movement = tf.convert_to_tensor(telemetry["movement_rate_by_chain"], tf.float64)
            repeated = tf.convert_to_tensor(telemetry["repeated_state_fraction_by_chain"], tf.float64)
            normalized = tf.convert_to_tensor(
                telemetry["normalized_return_displacement_by_chain"], tf.float64
            )
            path_return = tf.convert_to_tensor(
                telemetry["path_return_fraction_by_chain"], tf.float64
            )
            if not bool(
                tf.reduce_all(
                    tf.logical_and(
                        tf.logical_and(movement >= 0.05, repeated <= 0.95),
                        normalized >= 1.0e-4,
                    )
                ).numpy()
            ):
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
        wall = time.perf_counter() - started
        row = {
            "role": role,
            "num_leapfrog_steps": int(leapfrog),
            "step_size": float(epsilon),
            "seed": seed,
            "chain_means": chain_means,
            "hard_rejection_reasons": tuple(dict.fromkeys(reasons)),
            "native_divergence_status": divergence_status,
            "native_divergence_count": divergence_count,
            "telemetry": telemetry_payload,
            "wall_seconds": wall,
            "runner_metadata": result.metadata,
            "all_draws_discarded": True,
        }
        self.events.append(row)
        return row

    def primary(self, request: Any) -> Any:
        from bayesfilter.inference.hmc_operational_broad_grid import (
            OperationalPrimaryCandidate,
            classify_operational_pair_evidence,
            operational_broad_seed,
        )

        if request.mass_handoff_signature != self.handoff.signature:
            raise ValueError("primary request mass handoff changed")
        started = time.perf_counter()
        tune = self._tune_runner.run(
            seed=request.tune_seed,
            step_size=INITIAL_STEP_SIZE,
            num_leapfrog_steps=request.num_leapfrog_steps,
        )
        step_trace = self.tf.convert_to_tensor(tune.trace.get("step_size"), self.tf.float64)
        if step_trace.shape.rank not in {1, 2}:
            raise ValueError("dual averaging did not expose a scalar step trace")
        final_step = self.tf.reshape(step_trace[-1], [-1])
        if not bool(self.tf.reduce_all(self.tf.math.is_finite(final_step)).numpy()):
            raise ValueError("dual averaging produced nonfinite epsilon")
        spread = float((self.tf.reduce_max(final_step) - self.tf.reduce_min(final_step)).numpy())
        scale = max(1.0, float(self.tf.reduce_max(self.tf.abs(final_step)).numpy()))
        if spread > 1.0e-10 * scale:
            raise ValueError("dual averaging did not produce one common epsilon")
        epsilon = float(self.tf.reduce_mean(final_step).numpy())
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("dual averaging produced invalid epsilon")
        tune_health = self._combined_health(tune.samples)
        if tune_health:
            raise ValueError("dual-averaging target health failed: " + ",".join(tune_health))
        tune_event = {
            "role": "independent_epsilon_tune",
            "num_leapfrog_steps": request.num_leapfrog_steps,
            "initial_step_size": INITIAL_STEP_SIZE,
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
        evidence_payload = {
            "tune": tune_event,
            "screens": screens,
        }
        evidence = classify_operational_pair_evidence(
            chain_run_means=tuple(
                value for screen in screens for value in screen["chain_means"]
            ),
            evidence_signature=_signature(
                "bayesfilter.pp_ukf.primary_evidence.v1", evidence_payload
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
                "bayesfilter.pp_ukf.independent_epsilon_tune.v1", tune_event
            ),
        )

    def guard(self, request: Any) -> Any:
        from bayesfilter.inference.hmc_operational_broad_grid import (
            SameEpsilonNeighborGuard,
            classify_operational_pair_evidence,
        )

        if request.mass_handoff_signature != self.handoff.signature:
            raise ValueError("guard request mass handoff changed")
        screens = tuple(
            self._screen(
                leapfrog=request.num_leapfrog_steps,
                epsilon=request.inherited_step_size,
                seed=seed,
                role="same_epsilon_neighbor_guard_screen",
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
                "bayesfilter.pp_ukf.same_epsilon_guard_evidence.v1",
                {"request": request.payload(), "screens": screens},
            ),
            policy=self.policy,
            hard_rejection_reasons=reasons,
        )
        return SameEpsilonNeighborGuard(request=request, evidence=evidence)


def _build_handoff(
    *,
    adapter_signature: str,
    transport_manifest_hash: str,
    evidence_plan: str = PLAN,
) -> Any:
    from bayesfilter.inference.hmc_operational_broad_grid import OperationalMassHandoff

    identity_metric = _signature(
        "bayesfilter.pp_ukf.fixed_identity_metric.v1",
        {"dimension": 6, "coordinate": transport_manifest_hash},
    )
    coordinate = _signature(
        "bayesfilter.pp_ukf.frozen_transport_coordinate.v1",
        {"transport_manifest_hash": transport_manifest_hash},
    )
    lineage = _signature(
        "bayesfilter.pp_ukf.operational_broad_grid_lineage.v1",
        {
            "target_signature": TARGET_SIGNATURE,
            "transport_sha256": TRANSPORT_SHA256,
            "adapter_signature": adapter_signature,
            "coordinate_signature": coordinate,
            "metric_signature": identity_metric,
            "starts": (0.0, 0.1, -0.1, 0.16),
            "dtype": "float64",
            "jit_compile": True,
        },
    )
    return OperationalMassHandoff(
        update_disposition="fixed_identity",
        prior_metric_signature=identity_metric,
        frozen_metric_signature=identity_metric,
        coordinate_signature=coordinate,
        adapter_signature=adapter_signature,
        target_signature=TARGET_SIGNATURE,
        lineage_signature=lineage,
        canonical_covariance_signature=identity_metric,
        latent_metric_signature=identity_metric,
        metric_evidence_signature=_signature(
            "bayesfilter.pp_ukf.fixed_identity_policy_evidence.v1",
            {"plan": str(evidence_plan), "mass_updates": False},
        ),
        retained_prior_metric=True,
        latent_identity_equivalence_proven=True,
    )


def run_campaign(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output_root.exists():
        raise FileExistsError(f"output root must be fresh: {args.output_root}")
    args.output_root.mkdir(parents=True)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    started_utc = datetime.now(timezone.utc)
    started = time.perf_counter()

    import tensorflow as tf

    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.inference.hmc import stable_adapter_signature
    from bayesfilter.inference.hmc_operational_broad_grid import (
        OperationalBroadGridPolicy,
        assemble_operational_broad_grid_result,
        expand_same_epsilon_neighbor_guards,
        primary_requests,
    )
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    if _file_sha256(args.frozen_transport) != args.frozen_transport_sha256.lower():
        raise ValueError("frozen transport SHA-256 mismatch")
    if args.frozen_transport_sha256.lower() != TRANSPORT_SHA256:
        raise ValueError("campaign frozen transport differs from the reviewed artifact")
    loaded = load_frozen_neutra_artifact(
        json.loads(args.frozen_transport.read_text(encoding="utf-8")),
        expected_target_signature=TARGET_SIGNATURE,
    )
    bound = build_pp_ukf_bound_adapter()
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bound,
        transport=loaded.transport,
        target_scope="PP-UKF:operational_broad_fixed_identity_grid",
        evidence_path=PLAN,
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
    )
    adapter_signature = stable_adapter_signature(adapter)
    handoff = _build_handoff(
        adapter_signature=adapter_signature,
        transport_manifest_hash=adapter.transport_manifest_hash,
    )
    policy = OperationalBroadGridPolicy(
        root_seed=(20260721, 9300),
        confirmation_num_results=SCREEN_RESULTS,
        chain_count=CHAIN_COUNT,
        replication_count=REPLICATION_COUNT,
    )
    callbacks = PPUKFBroadGridCallbacks(
        tf=tf, adapter=adapter, policy=policy, handoff=handoff
    )
    requests = primary_requests(policy, handoff)
    primary_candidates = []
    primary_failures = []
    canary_started = time.perf_counter()
    try:
        primary_candidates.append(callbacks.primary(requests[0]))
    except Exception as error:  # noqa: BLE001 - terminal barrier evidence.
        primary_failures.append(f"{type(error).__name__}: {error}")
    canary_wall = time.perf_counter() - canary_started
    _write_progress_json(
        args.output_root / "progress.json",
        {
            "schema": "bayesfilter.pp_ukf.operational_broad_grid_progress.v1",
            "stage": "l3_primary_canary_complete",
            "completed_primary_count": len(primary_candidates),
            "planned_primary_count": len(requests),
            "primary_candidates": tuple(item.payload() for item in primary_candidates),
            "primary_failures": tuple(primary_failures),
            "completed_guard_count": 0,
            "events": callbacks.events,
            "elapsed_seconds": time.perf_counter() - started,
            "terminal": False,
        },
    )
    warm_screen_events = [
        item
        for item in callbacks.events
        if item.get("role") == "primary_independent_fixed_screen"
    ][-2:]
    if len(warm_screen_events) != 2:
        resource = {
            "full_grid_authorized": False,
            "stop_reason": "l3_canary_did_not_complete_two_warm_screen_calls",
            "canary_wall_seconds": canary_wall,
        }
    else:
        rate = max(
            float(item["wall_seconds"])
            / (PRIMARY_L_GRID[0] * (SCREEN_BURNIN + SCREEN_RESULTS))
            for item in warm_screen_events
        )
        resource = dict(
            project_remaining_campaign_seconds(
                warm_seconds_per_transition_leapfrog=rate,
                canary_wall_seconds=canary_wall,
                prior_charged_seconds=float(args.prior_charged_seconds),
            )
        )
        resource["stop_reason"] = (
            None
            if resource["full_grid_authorized"]
            else "projected_cumulative_work_exceeds_four_gpu_hour_cap"
        )
    _write_new_json(args.output_root / "primary_resource_decision.json", resource)

    if resource.get("full_grid_authorized") is True and not primary_failures:
        for request in requests[1:]:
            try:
                primary_candidates.append(callbacks.primary(request))
            except Exception as error:  # noqa: BLE001 - recorded barrier invalidity.
                primary_failures.append(f"L={request.num_leapfrog_steps}: {type(error).__name__}: {error}")
            _write_progress_json(
                args.output_root / "progress.json",
                {
                    "schema": "bayesfilter.pp_ukf.operational_broad_grid_progress.v1",
                    "stage": "primary_barrier_running",
                    "completed_primary_count": len(primary_candidates),
                    "planned_primary_count": len(requests),
                    "primary_candidates": tuple(
                        item.payload() for item in primary_candidates
                    ),
                    "primary_failures": tuple(primary_failures),
                    "completed_guard_count": 0,
                    "events": callbacks.events,
                    "elapsed_seconds": time.perf_counter() - started,
                    "terminal": False,
                },
            )
        guards = []
        guard_failures = []
        if not primary_failures:
            guard_requests = expand_same_epsilon_neighbor_guards(
                primary_candidates, policy=policy, handoff=handoff
            )
            completed_screen_events = [
                item
                for item in callbacks.events
                if str(item.get("role", "")).endswith("fixed_screen")
            ]
            observed_rate = max(
                float(item["wall_seconds"])
                / (
                    int(item["num_leapfrog_steps"])
                    * (SCREEN_BURNIN + SCREEN_RESULTS)
                )
                for item in completed_screen_events
            )
            guard_resource = project_guard_barrier_seconds(
                warm_seconds_per_transition_leapfrog=observed_rate,
                current_attempt_wall_seconds=time.perf_counter() - started,
                prior_charged_seconds=float(args.prior_charged_seconds),
                guard_l_values=tuple(
                    request.num_leapfrog_steps for request in guard_requests
                ),
            )
            _write_new_json(
                args.output_root / "guard_resource_decision.json", guard_resource
            )
            if guard_resource["guard_barrier_authorized"]:
                for request in guard_requests:
                    try:
                        guards.append(callbacks.guard(request))
                    except Exception as error:  # noqa: BLE001 - recorded barrier invalidity.
                        guard_failures.append(
                            f"L={request.num_leapfrog_steps}: {type(error).__name__}: {error}"
                        )
                    _write_progress_json(
                        args.output_root / "progress.json",
                        {
                            "schema": "bayesfilter.pp_ukf.operational_broad_grid_progress.v1",
                            "stage": "guard_barrier_running",
                            "completed_primary_count": len(primary_candidates),
                            "planned_primary_count": len(requests),
                            "primary_candidates": tuple(
                                item.payload() for item in primary_candidates
                            ),
                            "primary_failures": tuple(primary_failures),
                            "completed_guard_count": len(guards),
                            "planned_guard_count": len(guard_requests),
                            "guard_candidates": tuple(
                                item.payload() for item in guards
                            ),
                            "guard_failures": tuple(guard_failures),
                            "events": callbacks.events,
                            "elapsed_seconds": time.perf_counter() - started,
                            "terminal": False,
                        },
                    )
            elif guard_requests:
                guard_failures.append(
                    "resource_projection_stop_before_complete_guard_barrier"
                )
        else:
            guard_resource = {
                "stage": "actual_same_epsilon_neighbor_guard_barrier",
                "guard_barrier_authorized": False,
                "stop_reason": "primary_barrier_incomplete",
            }
        result = assemble_operational_broad_grid_result(
            policy=policy,
            handoff=handoff,
            primary_candidates=primary_candidates,
            guard_candidates=guards,
            primary_failure_reasons=primary_failures,
            guard_failure_reasons=guard_failures,
        )
        status = (
            "resource_projection_stop_before_complete_guard_barrier"
            if guard_failures
            == ["resource_projection_stop_before_complete_guard_barrier"]
            else result.disposition
        )
    else:
        result = assemble_operational_broad_grid_result(
            policy=policy,
            handoff=handoff,
            primary_candidates=primary_candidates,
            primary_failure_reasons=(
                tuple(primary_failures)
                if primary_failures
                else ("resource_projection_stop_before_complete_primary_barrier",)
            ),
        )
        status = (
            "l3_canary_failed"
            if primary_failures
            else "resource_projection_stop_before_complete_primary_barrier"
        )
        guard_resource = {
            "stage": "actual_same_epsilon_neighbor_guard_barrier",
            "guard_barrier_authorized": False,
            "stop_reason": "complete_primary_barrier_not_authorized",
        }
    wall = time.perf_counter() - started
    private_payload = {
        "schema": "bayesfilter.pp_ukf.operational_broad_grid_campaign.private.v1",
        "status": status,
        "grid": result.payload(),
        "events": callbacks.events,
        "resource_decision": resource,
        "guard_resource_decision": guard_resource,
        "all_tuning_draws_discarded": True,
    }
    public_payload = {
        "schema": "bayesfilter.pp_ukf.operational_broad_grid_campaign.public.v1",
        "status": status,
        "grid": result.public_payload(),
        "resource_decision": resource,
        "guard_resource_decision": guard_resource,
        "wall_seconds": wall,
        "statistical_ranking_supported": False,
        "retained_sampling_authorized": False,
        "nonclaims": result.public_payload()["nonclaims"],
    }
    _write_new_json(args.output_root / "private_result.json", private_payload)
    _write_new_json(args.output_root / "public_result.json", public_payload)
    try:
        allocator = {
            key + "_bytes": int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        }
    except (RuntimeError, ValueError):
        allocator = {"status": "unavailable"}
    manifest = {
        "schema": "bayesfilter.pp_ukf.operational_broad_grid_manifest.v1",
        "status": status,
        "started_utc": started_utc.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "devices": tuple(str(item) for item in tf.config.list_logical_devices()),
        "memory_policy": memory_policy,
        "gpu_allocator": allocator,
        "jit_compile": True,
        "tf32_execution_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "dtype": "float64",
        "target_signature": TARGET_SIGNATURE,
        "adapter_signature": adapter_signature,
        "transport_path": args.frozen_transport,
        "transport_sha256": TRANSPORT_SHA256,
        "metric_policy": "fixed_identity",
        "initial_step_size": INITIAL_STEP_SIZE,
        "initial_step_size_role": "prior_diagnostic_warm_start_only",
        "initial_step_size_source": WARM_START_SOURCE,
        "primary_l_grid": PRIMARY_L_GRID,
        "root_seed": policy.root_seed,
        "wall_seconds": wall,
        "output_paths": {
            "private_result": args.output_root / "private_result.json",
            "public_result": args.output_root / "public_result.json",
            "primary_resource_decision": args.output_root / "primary_resource_decision.json",
            "guard_resource_decision": (
                args.output_root / "guard_resource_decision.json"
                if (args.output_root / "guard_resource_decision.json").exists()
                else None
            ),
        },
        "plan_path": PLAN,
        "all_tuning_draws_discarded": True,
        "sampling_launched": False,
    }
    _write_new_json(args.output_root / "run_manifest.json", manifest)
    _write_progress_json(
        args.output_root / "progress.json",
        {
            "schema": "bayesfilter.pp_ukf.operational_broad_grid_progress.v1",
            "stage": "terminal",
            "status": status,
            "completed_primary_count": len(result.primary_candidates),
            "planned_primary_count": len(result.primary_barrier.planned_signatures),
            "completed_guard_count": len(result.guard_candidates),
            "planned_guard_count": len(result.guard_barrier.planned_signatures),
            "wall_seconds": wall,
            "terminal": True,
        },
    )
    return public_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frozen-transport", type=Path, required=True)
    parser.add_argument("--frozen-transport-sha256", default=TRANSPORT_SHA256)
    parser.add_argument("--prior-charged-seconds", type=float, default=PRIOR_CHARGED_SECONDS)
    args = parser.parse_args()
    result = run_campaign(args)
    print(json.dumps({"status": result["status"], "output_root": str(args.output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
