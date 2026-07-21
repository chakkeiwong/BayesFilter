#!/usr/bin/env python3
"""Run bounded state-continuing epsilon repair for PP-UKF broad-grid HMC."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

BASE_PATH = ROOT / "docs/benchmarks/run_pp_ukf_operational_broad_grid_20260721.py"
BASE_SPEC = importlib.util.spec_from_file_location("pp_ukf_broad_grid_base", BASE_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError("cannot load PP-UKF broad-grid base driver")
base = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base)

PLAN = "docs/plans/bayesfilter-pp-ukf-state-continuing-epsilon-repair-plan-2026-07-21.md"
PRIOR_RESULT = Path(
    "docs/plans/artifacts/bayesfilter-pp-ukf-operational-broad-grid-20260721/"
    "attempt-02/private_result.json"
)
TARGET_SIGNATURE = base.TARGET_SIGNATURE
TRANSPORT_SHA256 = base.TRANSPORT_SHA256
PRIMARY_L_GRID = base.PRIMARY_L_GRID
PRIOR_EPSILON_BY_L = {
    3: 0.8724049589170738,
    5: 0.8426345584765329,
    9: 0.7489709357241571,
    13: 0.69086551957137,
    18: 0.6813265222611998,
    25: 0.6800917535732008,
}
CHAIN_COUNT = 4
REPLICATION_COUNT = 3
ADAPTATION_STEPS = 96
POST_ADAPTATION_RESULTS = 32
CALIBRATION_RESULTS = 32
CALIBRATION_BURNIN = 1
CALIBRATION_REGION = (0.68, 0.72)
EPSILON_REPAIR_FACTOR = 1.20
MAX_EPSILON_REPAIRS = 3
FINAL_SCREEN_RESULTS = 96
FINAL_SCREEN_BURNIN = 8
CAMPAIGN_CAP_SECONDS = 4.0 * 3600.0
PRIOR_CHARGED_SECONDS = 3930.3757156120264 + 3063.6296786410094
PROJECTION_MARGIN = 1.50


def next_repair_epsilon(
    *,
    epsilon: float,
    acceptance_mean: float,
    lower_epsilon: float | None,
    upper_epsilon: float | None,
) -> tuple[float, float | None, float | None, str]:
    """Return the next epsilon and updated monotone acceptance bracket."""

    step = float(epsilon)
    mean = float(acceptance_mean)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not math.isfinite(mean) or not 0.0 <= mean <= 1.0:
        raise ValueError("acceptance_mean must lie inside [0, 1]")
    low = None if lower_epsilon is None else float(lower_epsilon)
    high = None if upper_epsilon is None else float(upper_epsilon)
    if mean > CALIBRATION_REGION[1]:
        low = step if low is None else max(low, step)
        direction = "increase_epsilon"
    elif mean < CALIBRATION_REGION[0]:
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
        proposal = step * EPSILON_REPAIR_FACTOR
    else:
        proposal = step / EPSILON_REPAIR_FACTOR
    return proposal, low, high, direction


def prospective_primary_projection(prior_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Project the complete repaired primary barrier from measured prior rates."""

    events = tuple(prior_payload.get("events", ()))
    rows = []
    for leapfrog in PRIMARY_L_GRID:
        tune = next(
            item
            for item in events
            if item.get("role") == "independent_epsilon_tune"
            and int(item.get("num_leapfrog_steps")) == leapfrog
        )
        screens = tuple(
            item
            for item in events
            if item.get("role") == "primary_independent_fixed_screen"
            and int(item.get("num_leapfrog_steps")) == leapfrog
        )
        if len(screens) != REPLICATION_COUNT:
            raise ValueError("prior result lacks complete per-L screen timing")
        tune_seconds = float(tune["wall_seconds"]) * (
            ADAPTATION_STEPS + POST_ADAPTATION_RESULTS
        ) / (base.TUNE_BURNIN + base.TUNE_RESULTS)
        maximum_screen_seconds = max(float(item["wall_seconds"]) for item in screens)
        repair_seconds = maximum_screen_seconds * (
            MAX_EPSILON_REPAIRS
            * (CALIBRATION_BURNIN + CALIBRATION_RESULTS)
            / (base.SCREEN_BURNIN + base.SCREEN_RESULTS)
        )
        final_seconds = maximum_screen_seconds * (
            REPLICATION_COUNT
            * (FINAL_SCREEN_BURNIN + FINAL_SCREEN_RESULTS)
            / (base.SCREEN_BURNIN + base.SCREEN_RESULTS)
        )
        rows.append(
            {
                "num_leapfrog_steps": leapfrog,
                "adaptation_and_post_seconds": tune_seconds,
                "maximum_repair_seconds": repair_seconds,
                "final_screen_seconds": final_seconds,
                "projected_seconds": tune_seconds + repair_seconds + final_seconds,
            }
        )
    unscaled = math.fsum(float(item["projected_seconds"]) for item in rows)
    projected = PROJECTION_MARGIN * unscaled
    cumulative = PRIOR_CHARGED_SECONDS + projected
    return {
        "schema": "bayesfilter.pp_ukf.state_continuing_primary_projection.v1",
        "rows": tuple(rows),
        "unscaled_projected_seconds": unscaled,
        "projection_margin": PROJECTION_MARGIN,
        "projected_new_seconds": projected,
        "prior_charged_seconds": PRIOR_CHARGED_SECONDS,
        "projected_cumulative_seconds": cumulative,
        "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
        "primary_barrier_authorized": cumulative <= CAMPAIGN_CAP_SECONDS,
    }


def _state_signature(tf: Any, state: Any) -> str:
    serialized = bytes(tf.io.serialize_tensor(tf.convert_to_tensor(state, tf.float64)).numpy())
    return hashlib.sha256(serialized).hexdigest()


class StateContinuingCallbacks(base.PPUKFBroadGridCallbacks):
    """Per-L adaptation, calibration repair, and disjoint final screens."""

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
        self._calibrated_states: dict[str, Any] = {}
        self._adapt_runner = build_reusable_full_chain_tfp_hmc_runner(
            adapter,
            self.initial_state,
            FullChainHMCConfig(
                num_results=POST_ADAPTATION_RESULTS,
                num_burnin_steps=ADAPTATION_STEPS,
                step_size=PRIOR_EPSILON_BY_L[3],
                num_leapfrog_steps=3,
                seed=policy.root_seed,
                use_xla=True,
                trace_policy="standard",
                target_status_trace_policy="none",
                tuning_policy=HMCTuningPolicy.fixed_mass_dual_averaging(
                    num_adaptation_steps=ADAPTATION_STEPS,
                    target_accept_prob=0.70,
                    source=PLAN,
                ),
                target_scope=adapter.target_scope,
            ),
            dynamic_num_leapfrog_steps=True,
        )
        self._calibration_runner = build_reusable_full_chain_tfp_hmc_runner(
            adapter,
            self.initial_state,
            FullChainHMCConfig(
                num_results=CALIBRATION_RESULTS,
                num_burnin_steps=CALIBRATION_BURNIN,
                step_size=PRIOR_EPSILON_BY_L[3],
                num_leapfrog_steps=3,
                seed=policy.root_seed,
                use_xla=True,
                trace_policy="standard",
                target_status_trace_policy="none",
                target_scope=adapter.target_scope,
            ),
            dynamic_num_leapfrog_steps=True,
        )
        self._final_screen_runner = build_reusable_full_chain_tfp_hmc_runner(
            adapter,
            self.initial_state,
            FullChainHMCConfig(
                num_results=FINAL_SCREEN_RESULTS,
                num_burnin_steps=FINAL_SCREEN_BURNIN,
                step_size=PRIOR_EPSILON_BY_L[3],
                num_leapfrog_steps=3,
                seed=policy.root_seed,
                use_xla=True,
                trace_policy="standard",
                target_status_trace_policy="none",
                target_scope=adapter.target_scope,
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
        from bayesfilter.inference.hmc_verification import summarize_hmc_tuning_telemetry

        tf = self.tf
        reasons = list(self._combined_health(result.samples))
        trace = result.trace
        required = ("log_accept_ratio", "is_accepted", "target_log_prob")
        if any(name not in trace for name in required):
            raise ValueError("required standard HMC trace is missing")
        telemetry = summarize_hmc_tuning_telemetry(
            samples=result.samples,
            log_accept_ratio=trace["log_accept_ratio"],
            is_accepted=trace["is_accepted"],
        )
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
        repeated = tf.convert_to_tensor(
            telemetry["repeated_state_fraction_by_chain"], tf.float64
        )
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
        final_state = tf.convert_to_tensor(result.samples[-1], tf.float64)
        row = {
            "role": role,
            "num_leapfrog_steps": int(leapfrog),
            "step_size": float(epsilon),
            "seed": seed,
            "chain_means": chain_means,
            "grand_mean": math.fsum(chain_means) / len(chain_means),
            "hard_rejection_reasons": tuple(dict.fromkeys(reasons)),
            "native_divergence_status": divergence_status,
            "native_divergence_count": divergence_count,
            "telemetry": base._json_ready(telemetry),
            "wall_seconds": float(wall_seconds),
            "runner_metadata": result.metadata,
            "initial_state_signature": initial_state_signature,
            "final_state_signature": _state_signature(tf, final_state),
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
        initial_signature = _state_signature(self.tf, current_state)
        started = time.perf_counter()
        result = runner.run(
            current_state=current_state,
            seed=seed,
            step_size=epsilon,
            num_leapfrog_steps=leapfrog,
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
        from bayesfilter.inference.hmc_operational_broad_grid import (
            operational_broad_seed,
        )

        if request.mass_handoff_signature != self.handoff.signature:
            raise ValueError("primary request mass handoff changed")
        leapfrog = request.num_leapfrog_steps
        initial_epsilon = PRIOR_EPSILON_BY_L[leapfrog]
        initial_signature = _state_signature(self.tf, self.initial_state)
        started = time.perf_counter()
        adapted = self._adapt_runner.run(
            current_state=self.initial_state,
            seed=request.tune_seed,
            step_size=initial_epsilon,
            num_leapfrog_steps=leapfrog,
        )
        step_trace = self.tf.reshape(
            self.tf.convert_to_tensor(adapted.trace.get("step_size"), self.tf.float64),
            [-1],
        )
        if int(step_trace.shape[0]) != POST_ADAPTATION_RESULTS:
            raise ValueError("post-adaptation step trace has the wrong length")
        if not bool(self.tf.reduce_all(self.tf.math.is_finite(step_trace)).numpy()):
            raise ValueError("dual averaging produced nonfinite epsilon")
        epsilon = float(step_trace[-1].numpy())
        if not bool(
            self.tf.reduce_all(
                self.tf.equal(step_trace, self.tf.fill(self.tf.shape(step_trace), step_trace[-1]))
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
        lower_epsilon = None
        upper_epsilon = None
        for repair_index in range(MAX_EPSILON_REPAIRS):
            if CALIBRATION_REGION[0] <= acceptance_mean <= CALIBRATION_REGION[1]:
                break
            next_epsilon, lower_epsilon, upper_epsilon, action = next_repair_epsilon(
                epsilon=epsilon,
                acceptance_mean=acceptance_mean,
                lower_epsilon=lower_epsilon,
                upper_epsilon=upper_epsilon,
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
        calibrated_state_signature = _state_signature(self.tf, current_state)
        tune_event = {
            "role": "state_continuing_epsilon_tune",
            "num_leapfrog_steps": leapfrog,
            "initial_step_size": initial_epsilon,
            "tuned_step_size": epsilon,
            "target_accept_prob": 0.70,
            "calibration_region": CALIBRATION_REGION,
            "calibration_final_mean": acceptance_mean,
            "calibration_region_reached": (
                CALIBRATION_REGION[0] <= acceptance_mean <= CALIBRATION_REGION[1]
            ),
            "calibration_rows": tuple(calibration_rows),
            "calibrated_state_signature": calibrated_state_signature,
            "state_continuation_performed": True,
            "all_draws_discarded": True,
        }
        self.events.append(tune_event)
        return epsilon, current_state, tune_event

    def primary(self, request: Any) -> Any:
        from bayesfilter.inference.hmc_operational_broad_grid import (
            OperationalPrimaryCandidate,
            classify_operational_pair_evidence,
            operational_broad_seed,
        )

        leapfrog = request.num_leapfrog_steps
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
            evidence_signature=base._signature(
                "bayesfilter.pp_ukf.state_continuing_primary_evidence.v1",
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
            tune_evidence_signature=base._signature(
                "bayesfilter.pp_ukf.state_continuing_epsilon_tune.v1", tune_event
            ),
        )
        self._calibrated_states[candidate.signature] = self.tf.identity(current_state)
        return candidate

    def guard(self, request: Any) -> Any:
        from bayesfilter.inference.hmc_operational_broad_grid import (
            SameEpsilonNeighborGuard,
            classify_operational_pair_evidence,
        )

        parent_signatures = tuple(sorted(request.parent_candidate_signatures))
        if len(parent_signatures) != 1 or any(
            signature not in self._calibrated_states for signature in parent_signatures
        ):
            raise ValueError("guard requires exactly one calibrated parent state")
        parent_signature = parent_signatures[0]
        current_state = self._calibrated_states[parent_signature]
        screens = tuple(
            self._fixed_run(
                runner=self._final_screen_runner,
                current_state=current_state,
                leapfrog=request.num_leapfrog_steps,
                epsilon=request.inherited_step_size,
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
            evidence_signature=base._signature(
                "bayesfilter.pp_ukf.state_continuing_guard_evidence.v1",
                {
                    "request": request.payload(),
                    "parent_calibrated_state_signature": _state_signature(
                        self.tf, current_state
                    ),
                    "screens": screen_payloads,
                },
            ),
            policy=self.policy,
            hard_rejection_reasons=reasons,
        )
        return SameEpsilonNeighborGuard(request=request, evidence=evidence)


def _progress_payload(
    *,
    stage: str,
    callbacks: StateContinuingCallbacks,
    primaries: Sequence[Any],
    primary_failures: Sequence[str],
    guards: Sequence[Any] = (),
    guard_failures: Sequence[str] = (),
    started: float,
    terminal: bool = False,
) -> Mapping[str, Any]:
    return {
        "schema": "bayesfilter.pp_ukf.state_continuing_epsilon_repair_progress.v1",
        "stage": stage,
        "completed_primary_count": len(primaries),
        "planned_primary_count": len(PRIMARY_L_GRID),
        "primary_candidates": tuple(item.payload() for item in primaries),
        "primary_failures": tuple(primary_failures),
        "completed_guard_count": len(guards),
        "guard_candidates": tuple(item.payload() for item in guards),
        "guard_failures": tuple(guard_failures),
        "events": callbacks.events,
        "elapsed_seconds": time.perf_counter() - started,
        "terminal": terminal,
    }


def run_campaign(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output_root.exists():
        raise FileExistsError(f"output root must be fresh: {args.output_root}")
    args.output_root.mkdir(parents=True)
    prior_payload = json.loads((ROOT / PRIOR_RESULT).read_text(encoding="utf-8"))
    projection = prospective_primary_projection(prior_payload)
    base._write_new_json(args.output_root / "primary_resource_decision.json", projection)
    if projection["primary_barrier_authorized"] is not True:
        raise RuntimeError("repaired primary barrier exceeds the unchanged campaign cap")

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
    if base._file_sha256(args.frozen_transport) != TRANSPORT_SHA256:
        raise ValueError("frozen transport SHA-256 mismatch")
    loaded = load_frozen_neutra_artifact(
        json.loads(args.frozen_transport.read_text(encoding="utf-8")),
        expected_target_signature=TARGET_SIGNATURE,
    )
    bound = base.build_pp_ukf_bound_adapter()
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bound,
        transport=loaded.transport,
        target_scope="PP-UKF:state_continuing_epsilon_repair",
        evidence_path=PLAN,
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
    )
    adapter_signature = stable_adapter_signature(adapter)
    handoff = base._build_handoff(
        adapter_signature=adapter_signature,
        transport_manifest_hash=adapter.transport_manifest_hash,
        evidence_plan=PLAN,
    )
    policy = OperationalBroadGridPolicy(
        root_seed=(20260721, 9400),
        confirmation_num_results=FINAL_SCREEN_RESULTS,
        chain_count=CHAIN_COUNT,
        replication_count=REPLICATION_COUNT,
    )
    callbacks = StateContinuingCallbacks(
        tf=tf, adapter=adapter, policy=policy, handoff=handoff
    )
    requests = primary_requests(policy, handoff)
    projected_row_by_l = {
        int(item["num_leapfrog_steps"]): item for item in projection["rows"]
    }
    primaries = []
    primary_failures = []
    for request in requests:
        next_arm_projection = (
            PROJECTION_MARGIN
            * float(projected_row_by_l[request.num_leapfrog_steps]["projected_seconds"])
        )
        if (
            PRIOR_CHARGED_SECONDS
            + (time.perf_counter() - started)
            + next_arm_projection
            > CAMPAIGN_CAP_SECONDS
        ):
            primary_failures.append(
                f"L={request.num_leapfrog_steps}: resource_projection_stop_before_primary"
            )
            break
        try:
            primaries.append(callbacks.primary(request))
        except Exception as error:  # noqa: BLE001 - barrier-invalidity receipt.
            primary_failures.append(
                f"L={request.num_leapfrog_steps}: {type(error).__name__}: {error}"
            )
        base._write_progress_json(
            args.output_root / "progress.json",
            _progress_payload(
                stage="primary_barrier_running",
                callbacks=callbacks,
                primaries=primaries,
                primary_failures=primary_failures,
                started=started,
            ),
        )
    guards = []
    guard_failures = []
    guard_projection: Mapping[str, Any]
    if not primary_failures:
        guard_requests = expand_same_epsilon_neighbor_guards(
            primaries, policy=policy, handoff=handoff
        )
        completed_screens = tuple(
            item
            for item in callbacks.events
            if item.get("role") == "state_continuing_primary_fresh_screen"
        )
        maximum_seconds_per_leapfrog_transition = max(
            float(item["wall_seconds"])
            / (
                int(item["num_leapfrog_steps"])
                * (FINAL_SCREEN_BURNIN + FINAL_SCREEN_RESULTS)
            )
            for item in completed_screens
        )
        guard_work = sum(
            request.num_leapfrog_steps
            * REPLICATION_COUNT
            * (FINAL_SCREEN_BURNIN + FINAL_SCREEN_RESULTS)
            for request in guard_requests
        )
        projected_guard_seconds = (
            PROJECTION_MARGIN * maximum_seconds_per_leapfrog_transition * guard_work
        )
        projected_cumulative = (
            PRIOR_CHARGED_SECONDS
            + (time.perf_counter() - started)
            + projected_guard_seconds
        )
        guard_projection = {
            "schema": "bayesfilter.pp_ukf.state_continuing_guard_projection.v1",
            "guard_l_values": tuple(
                request.num_leapfrog_steps for request in guard_requests
            ),
            "actual_guard_count": len(guard_requests),
            "guard_transition_leapfrogs": guard_work,
            "maximum_seconds_per_leapfrog_transition": (
                maximum_seconds_per_leapfrog_transition
            ),
            "projection_margin": PROJECTION_MARGIN,
            "projected_guard_seconds": projected_guard_seconds,
            "prior_charged_seconds": PRIOR_CHARGED_SECONDS,
            "current_attempt_wall_seconds": time.perf_counter() - started,
            "projected_cumulative_seconds": projected_cumulative,
            "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
            "guard_barrier_authorized": projected_cumulative <= CAMPAIGN_CAP_SECONDS,
        }
        base._write_new_json(
            args.output_root / "guard_resource_decision.json", guard_projection
        )
        if guard_projection["guard_barrier_authorized"]:
            for request in guard_requests:
                try:
                    guards.append(callbacks.guard(request))
                except Exception as error:  # noqa: BLE001 - barrier receipt.
                    guard_failures.append(
                        f"L={request.num_leapfrog_steps}: {type(error).__name__}: {error}"
                    )
                base._write_progress_json(
                    args.output_root / "progress.json",
                    _progress_payload(
                        stage="guard_barrier_running",
                        callbacks=callbacks,
                        primaries=primaries,
                        primary_failures=primary_failures,
                        guards=guards,
                        guard_failures=guard_failures,
                        started=started,
                    ),
                )
        elif guard_requests:
            guard_failures.append("resource_projection_stop_before_guard_barrier")
    else:
        guard_projection = {
            "guard_barrier_authorized": False,
            "stop_reason": "primary_barrier_incomplete",
        }
    result = assemble_operational_broad_grid_result(
        policy=policy,
        handoff=handoff,
        primary_candidates=primaries,
        guard_candidates=guards,
        primary_failure_reasons=primary_failures,
        guard_failure_reasons=guard_failures,
    )
    status = (
        "resource_projection_stop_before_guard_barrier"
        if guard_failures == ["resource_projection_stop_before_guard_barrier"]
        else result.disposition
    )
    wall = time.perf_counter() - started
    private_payload = {
        "schema": "bayesfilter.pp_ukf.state_continuing_epsilon_repair.private.v1",
        "status": status,
        "grid": result.payload(),
        "events": callbacks.events,
        "primary_resource_decision": projection,
        "guard_resource_decision": guard_projection,
        "prior_result_role": "warm_starts_and_cost_observations_only",
        "prior_evidence_reused": False,
        "all_tuning_draws_discarded": True,
    }
    public_payload = {
        "schema": "bayesfilter.pp_ukf.state_continuing_epsilon_repair.public.v1",
        "status": status,
        "grid": result.public_payload(),
        "primary_resource_decision": projection,
        "guard_resource_decision": guard_projection,
        "wall_seconds": wall,
        "retained_sampling_authorized": False,
        "statistical_ranking_supported": False,
        "nonclaims": result.public_payload()["nonclaims"],
    }
    base._write_new_json(args.output_root / "private_result.json", private_payload)
    base._write_new_json(args.output_root / "public_result.json", public_payload)
    try:
        allocator = {
            key + "_bytes": int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        }
    except (RuntimeError, ValueError):
        allocator = {"status": "unavailable"}
    manifest = {
        "schema": "bayesfilter.pp_ukf.state_continuing_epsilon_repair_manifest.v1",
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
        "primary_l_grid": PRIMARY_L_GRID,
        "root_seed": policy.root_seed,
        "prior_charged_seconds": PRIOR_CHARGED_SECONDS,
        "wall_seconds": wall,
        "cumulative_charged_seconds": PRIOR_CHARGED_SECONDS + wall,
        "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
        "plan_path": PLAN,
        "all_tuning_draws_discarded": True,
        "sampling_launched": False,
    }
    base._write_new_json(args.output_root / "run_manifest.json", manifest)
    base._write_progress_json(
        args.output_root / "progress.json",
        {
            "schema": "bayesfilter.pp_ukf.state_continuing_epsilon_repair_progress.v1",
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
    args = parser.parse_args()
    result = run_campaign(args)
    print(
        json.dumps(
            {"status": result["status"], "output_root": str(args.output_root)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
