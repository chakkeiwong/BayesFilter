#!/usr/bin/env python3
"""Reclassify PP-UKF primaries and run exact-epsilon coverage probes."""

from __future__ import annotations

import argparse
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

REPAIR_PATH = ROOT / "docs/benchmarks/run_pp_ukf_state_continuing_epsilon_repair_20260721.py"
REPAIR_SPEC = importlib.util.spec_from_file_location(
    "pp_ukf_state_continuing_repair", REPAIR_PATH
)
if REPAIR_SPEC is None or REPAIR_SPEC.loader is None:
    raise RuntimeError("cannot load PP-UKF state-continuing repair driver")
repair = importlib.util.module_from_spec(REPAIR_SPEC)
REPAIR_SPEC.loader.exec_module(repair)
base = repair.base

PLAN = (
    "docs/plans/"
    "bayesfilter-pp-ukf-statistical-compatibility-and-guard-repair-plan-2026-07-21.md"
)
SOURCE_ROOT = Path(
    "docs/plans/artifacts/"
    "bayesfilter-pp-ukf-state-continuing-epsilon-repair-20260721/attempt-01"
)
SOURCE_PRIVATE = SOURCE_ROOT / "private_result.json"
SOURCE_MANIFEST = SOURCE_ROOT / "run_manifest.json"
CAMPAIGN_CAP_SECONDS = repair.CAMPAIGN_CAP_SECONDS
PROJECTION_MARGIN = 1.25
RUNNER_COMPILE_ALLOWANCE_SECONDS = 60.0
EXPECTED_NEXT_ROUND_L_VALUES = (5, 9, 12, 13, 14, 17, 18, 19, 24, 25)


def build_pp_ukf_frozen_validation_candidates(
    next_round_candidates: Sequence[Any],
    *,
    model_id: str,
    target_signature: str,
    tuning_scope_signature: str,
    validation_seed_root: tuple[int, int] = (20260722, 5100),
) -> tuple[Any, ...]:
    """Convert PP-UKF next-round records into generic validation candidates.

    This is an adapter seam only. It does not execute HMC or select a winner.
    Primary records retain their tuned epsilon; coverage records retain the
    exact inherited parent epsilon and parent candidate signature.
    """

    from bayesfilter.inference.frozen_kernel_validation import (
        FrozenValidationCandidate,
    )
    from bayesfilter.inference.hmc_operational_broad_grid import (
        OperationalPrimaryCandidate,
        SameEpsilonNeighborGuard,
        operational_broad_seed,
    )

    root = tuple(int(item) for item in validation_seed_root)
    if len(root) != 2 or any(item < 0 for item in root):
        raise ValueError("validation_seed_root must contain two nonnegative integers")
    converted = []
    for item in next_round_candidates:
        if isinstance(item, OperationalPrimaryCandidate):
            leapfrog = item.request.num_leapfrog_steps
            epsilon = item.tuned_step_size
            role = "independently_tuned"
            parent_id = None
            inherited_keys = ()
        elif isinstance(item, SameEpsilonNeighborGuard):
            leapfrog = item.request.num_leapfrog_steps
            epsilon = item.request.inherited_step_size
            role = "inherited_exact_one_hop_coverage"
            parent_id = item.request.parent_candidate_signatures[0]
            inherited_keys = ("step_size",)
        else:
            raise TypeError("next_round_candidates contains an unsupported record")
        converted.append(
            FrozenValidationCandidate(
                candidate_id=item.signature,
                model_id=model_id,
                target_signature=target_signature,
                tuning_scope_signature=tuning_scope_signature,
                controls={
                    "num_leapfrog_steps": leapfrog,
                    "step_size": epsilon,
                },
                control_provenance=role,
                execution_seed=operational_broad_seed(
                    root,
                    domain="generic_frozen_validation",
                    num_leapfrog_steps=leapfrog,
                    epsilon=epsilon,
                ),
                parent_candidate_id=parent_id,
                inherited_control_keys=inherited_keys,
            )
        )
    return tuple(converted)


def _source_classification_rows(
    source_payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    from bayesfilter.inference.hmc_operational_broad_grid import (
        OperationalBroadGridPolicy,
        classify_operational_pair_evidence,
    )

    policy = OperationalBroadGridPolicy(
        root_seed=(20260721, 9400),
        confirmation_num_results=repair.FINAL_SCREEN_RESULTS,
        chain_count=repair.CHAIN_COUNT,
        replication_count=repair.REPLICATION_COUNT,
    )
    rows = []
    for candidate in source_payload["grid"]["primary_candidates"]:
        leapfrog = int(candidate["request"]["num_leapfrog_steps"])
        source_evidence = candidate["evidence"]
        evidence = classify_operational_pair_evidence(
            chain_run_means=source_evidence["chain_run_means"],
            evidence_signature=source_evidence["evidence_signature"],
            policy=policy,
            hard_rejection_reasons=source_evidence["hard_rejection_reasons"],
        )
        if not math.isclose(
            evidence.grand_mean,
            float(source_evidence["grand_mean"]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"L={leapfrog} source point mean did not reproduce")
        rows.append(
            {
                "num_leapfrog_steps": leapfrog,
                "tuned_step_size": float(candidate["tuned_step_size"]),
                "source_disposition": source_evidence["disposition"],
                "corrected_evidence": evidence.payload(),
                "statistically_compatible": evidence.viable,
                "source_evidence_signature": source_evidence["evidence_signature"],
            }
        )
    if tuple(item["num_leapfrog_steps"] for item in rows) != repair.PRIMARY_L_GRID:
        raise ValueError("source primary grid is incomplete or reordered")
    return tuple(rows)


def prospective_guard_projection(
    source_payload: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    from bayesfilter.inference.hmc_operational_broad_grid import MAX_L, MIN_GUARD_L

    rows = _source_classification_rows(source_payload)
    compatible_l = tuple(
        int(item["num_leapfrog_steps"])
        for item in rows
        if item["statistically_compatible"]
    )
    events = tuple(source_payload["events"])
    reconstruction_seconds = math.fsum(
        float(event["wall_seconds"])
        for event in events
        if int(event.get("num_leapfrog_steps", -1)) in compatible_l
        and event.get("role")
        in {
            "state_continuing_adaptation_calibration",
            "state_continuing_epsilon_repair_calibration",
        }
    )
    screen_events = tuple(
        event
        for event in events
        if event.get("role") == "state_continuing_primary_fresh_screen"
    )
    if len(screen_events) != len(repair.PRIMARY_L_GRID) * repair.REPLICATION_COUNT:
        raise ValueError("source artifact lacks complete final-screen timing")
    steady_events = screen_events[1:]
    maximum_steady_seconds_per_leapfrog_transition = max(
        float(event["wall_seconds"])
        / (
            int(event["num_leapfrog_steps"])
            * (repair.FINAL_SCREEN_BURNIN + repair.FINAL_SCREEN_RESULTS)
        )
        for event in steady_events
    )
    coverage_probe_l_values = tuple(
        neighbor
        for parent in compatible_l
        for neighbor in (parent - 1, parent + 1)
        if MIN_GUARD_L <= neighbor <= MAX_L
    )
    guard_work = math.fsum(
        leapfrog
        * repair.REPLICATION_COUNT
        * (repair.FINAL_SCREEN_BURNIN + repair.FINAL_SCREEN_RESULTS)
        for leapfrog in coverage_probe_l_values
    )
    unscaled = (
        reconstruction_seconds
        + RUNNER_COMPILE_ALLOWANCE_SECONDS
        + maximum_steady_seconds_per_leapfrog_transition * guard_work
    )
    projected_new = PROJECTION_MARGIN * unscaled
    prior_charged = float(source_manifest["cumulative_charged_seconds"])
    cumulative = prior_charged + projected_new
    return {
        "schema": "bayesfilter.pp_ukf.statistical_compatibility_guard_projection.v2",
        "compatible_primary_l_values": compatible_l,
        "coverage_probe_l_values": coverage_probe_l_values,
        # Historical aliases remain readable; these are coverage probes, not
        # parent-promotion guard/veto values.
        "guard_l_values": coverage_probe_l_values,
        "guard_count": len(coverage_probe_l_values),
        "guard_transition_leapfrogs": guard_work,
        "reconstruction_seconds": reconstruction_seconds,
        "maximum_steady_seconds_per_leapfrog_transition": (
            maximum_steady_seconds_per_leapfrog_transition
        ),
        "runner_compile_allowance_seconds": RUNNER_COMPILE_ALLOWANCE_SECONDS,
        "projection_margin": PROJECTION_MARGIN,
        "unscaled_projected_seconds": unscaled,
        "projected_new_seconds": projected_new,
        "prior_charged_seconds": prior_charged,
        "projected_cumulative_seconds": cumulative,
        "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
        "guard_campaign_authorized": cumulative <= CAMPAIGN_CAP_SECONDS,
        "coverage_campaign_authorized": cumulative <= CAMPAIGN_CAP_SECONDS,
    }


def _corrected_primary_candidates(
    *,
    source_payload: Mapping[str, Any],
    policy: Any,
    handoff: Any,
) -> tuple[Any, ...]:
    from bayesfilter.inference.hmc_operational_broad_grid import (
        OperationalPrimaryCandidate,
        classify_operational_pair_evidence,
        primary_requests,
    )

    source_by_l = {
        int(item["request"]["num_leapfrog_steps"]): item
        for item in source_payload["grid"]["primary_candidates"]
    }
    candidates = []
    for request in primary_requests(policy, handoff):
        source = source_by_l[request.num_leapfrog_steps]
        source_evidence = source["evidence"]
        evidence = classify_operational_pair_evidence(
            chain_run_means=source_evidence["chain_run_means"],
            evidence_signature=source_evidence["evidence_signature"],
            policy=policy,
            hard_rejection_reasons=source_evidence["hard_rejection_reasons"],
        )
        candidates.append(
            OperationalPrimaryCandidate(
                request=request,
                tuned_step_size=float(source["tuned_step_size"]),
                evidence=evidence,
                metric_signature=handoff.frozen_metric_signature,
                coordinate_signature=handoff.coordinate_signature,
                lineage_signature=handoff.lineage_signature,
                tune_evidence_signature=source["tune_evidence_signature"],
            )
        )
    return tuple(candidates)


def _progress_payload(
    *,
    started: float,
    reconstruction_receipts: Sequence[Mapping[str, Any]],
    guards: Sequence[Any],
    planned_guard_count: int,
    guard_failures: Sequence[str] = (),
    terminal: bool = False,
) -> Mapping[str, Any]:
    return {
        "schema": "bayesfilter.pp_ukf.statistical_compatibility_guard_progress.v1",
        "stage": "terminal" if terminal else "guard_repair_running",
        "reconstructed_parent_count": len(reconstruction_receipts),
        "reconstruction_receipts": tuple(reconstruction_receipts),
        "completed_guard_count": len(guards),
        "planned_guard_count": planned_guard_count,
        "guard_candidates": tuple(item.payload() for item in guards),
        "guard_failures": tuple(guard_failures),
        "elapsed_seconds": time.perf_counter() - started,
        "terminal": terminal,
    }


def run_campaign(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output_root.exists():
        raise FileExistsError(f"output root must be fresh: {args.output_root}")
    args.output_root.mkdir(parents=True)
    source_payload = json.loads((ROOT / SOURCE_PRIVATE).read_text(encoding="utf-8"))
    source_manifest = json.loads((ROOT / SOURCE_MANIFEST).read_text(encoding="utf-8"))
    projection = prospective_guard_projection(source_payload, source_manifest)
    base._write_new_json(args.output_root / "resource_decision.json", projection)
    if projection["guard_campaign_authorized"] is not True:
        raise RuntimeError("corrected guard campaign exceeds unchanged budget")

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
    )
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    if base._file_sha256(args.frozen_transport) != repair.TRANSPORT_SHA256:
        raise ValueError("frozen transport SHA-256 mismatch")
    loaded = load_frozen_neutra_artifact(
        json.loads(args.frozen_transport.read_text(encoding="utf-8")),
        expected_target_signature=repair.TARGET_SIGNATURE,
    )
    bound = base.build_pp_ukf_bound_adapter()
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bound,
        transport=loaded.transport,
        target_scope="PP-UKF:state_continuing_epsilon_repair",
        evidence_path=repair.PLAN,
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
    )
    adapter_signature = stable_adapter_signature(adapter)
    if adapter_signature != source_manifest["adapter_signature"]:
        raise ValueError("source adapter signature drifted")
    handoff = base._build_handoff(
        adapter_signature=adapter_signature,
        transport_manifest_hash=adapter.transport_manifest_hash,
        evidence_plan=repair.PLAN,
    )
    policy = OperationalBroadGridPolicy(
        root_seed=(20260721, 9400),
        confirmation_num_results=repair.FINAL_SCREEN_RESULTS,
        chain_count=repair.CHAIN_COUNT,
        replication_count=repair.REPLICATION_COUNT,
    )
    primaries = _corrected_primary_candidates(
        source_payload=source_payload,
        policy=policy,
        handoff=handoff,
    )
    compatible = tuple(item for item in primaries if item.viable)
    if tuple(item.request.num_leapfrog_steps for item in compatible) != tuple(
        projection["compatible_primary_l_values"]
    ):
        raise ValueError("runtime compatible-primary set changed")

    callbacks = repair.StateContinuingCallbacks(
        tf=tf, adapter=adapter, policy=policy, handoff=handoff
    )
    source_tune_by_l = {
        int(item["num_leapfrog_steps"]): item
        for item in source_payload["events"]
        if item.get("role") == "state_continuing_epsilon_tune"
    }
    reconstruction_receipts = []
    for candidate in compatible:
        epsilon, state, tune_event = callbacks.calibrate_primary(candidate.request)
        source_tune = source_tune_by_l[candidate.request.num_leapfrog_steps]
        if epsilon.hex() != float(candidate.tuned_step_size).hex():
            raise ValueError(
                f"L={candidate.request.num_leapfrog_steps} reconstructed epsilon changed"
            )
        if (
            tune_event["calibrated_state_signature"]
            != source_tune["calibrated_state_signature"]
        ):
            raise ValueError(
                f"L={candidate.request.num_leapfrog_steps} calibrated state changed"
            )
        callbacks._calibrated_states[candidate.signature] = tf.identity(state)
        reconstruction_receipts.append(
            {
                "num_leapfrog_steps": candidate.request.num_leapfrog_steps,
                "tuned_step_size": epsilon,
                "epsilon_exact_match": True,
                "calibrated_state_signature": tune_event[
                    "calibrated_state_signature"
                ],
                "calibrated_state_exact_match": True,
            }
        )

    guard_requests = expand_same_epsilon_neighbor_guards(
        primaries, policy=policy, handoff=handoff
    )
    if tuple(item.num_leapfrog_steps for item in guard_requests) != tuple(
        projection["coverage_probe_l_values"]
    ):
        raise ValueError("runtime guard set changed")
    guards = []
    guard_failures = []
    for index, request in enumerate(guard_requests):
        remaining_work = math.fsum(
            item.num_leapfrog_steps
            * repair.REPLICATION_COUNT
            * (repair.FINAL_SCREEN_BURNIN + repair.FINAL_SCREEN_RESULTS)
            for item in guard_requests[index:]
        )
        projected_remaining = (
            PROJECTION_MARGIN
            * float(projection["maximum_steady_seconds_per_leapfrog_transition"])
            * remaining_work
        )
        if (
            float(source_manifest["cumulative_charged_seconds"])
            + (time.perf_counter() - started)
            + projected_remaining
            > CAMPAIGN_CAP_SECONDS
        ):
            guard_failures.append("resource_projection_stop_before_guard_completion")
            break
        try:
            guards.append(callbacks.guard(request))
        except Exception as error:  # noqa: BLE001 - terminal barrier receipt.
            guard_failures.append(
                f"L={request.num_leapfrog_steps}: {type(error).__name__}: {error}"
            )
        base._write_progress_json(
            args.output_root / "progress.json",
            _progress_payload(
                started=started,
                reconstruction_receipts=reconstruction_receipts,
                guards=guards,
                planned_guard_count=len(guard_requests),
                guard_failures=guard_failures,
            ),
        )

    result = assemble_operational_broad_grid_result(
        policy=policy,
        handoff=handoff,
        primary_candidates=primaries,
        guard_candidates=guards,
        guard_failure_reasons=guard_failures,
    )
    expected_next_round = tuple(
        sorted(
            {
                item.request.num_leapfrog_steps
                for item in result.next_round_candidates
            }
        )
    )
    if expected_next_round != result.next_round_l_values:
        raise ValueError("next-round L values are not a unique candidate union")
    if result.guard_barrier.complete and expected_next_round != EXPECTED_NEXT_ROUND_L_VALUES:
        raise ValueError(
            "PP-UKF next-round set changed: "
            f"expected {EXPECTED_NEXT_ROUND_L_VALUES}, got {expected_next_round}"
        )
    wall = time.perf_counter() - started
    status = result.disposition
    private_payload = {
        "schema": "bayesfilter.pp_ukf.statistical_compatibility_guard_repair.private.v2",
        "status": status,
        "grid": result.payload(),
        "source_primary_artifact": SOURCE_PRIVATE,
        "source_primary_artifact_sha256": base._file_sha256(ROOT / SOURCE_PRIVATE),
        "source_primary_reclassified_without_rerun": True,
        "reconstruction_receipts": tuple(reconstruction_receipts),
        "events": callbacks.events,
        "resource_decision": projection,
        "all_tuning_draws_discarded": True,
        "next_round_l_values": result.next_round_l_values,
        "next_round_selection": "compatible_primaries_union_compatible_one_hop_coverage",
        "next_round_ranking_performed": False,
    }
    public_payload = {
        "schema": "bayesfilter.pp_ukf.statistical_compatibility_guard_repair.public.v2",
        "status": status,
        "grid": result.public_payload(),
        "resource_decision": projection,
        "wall_seconds": wall,
        "retained_sampling_authorized": False,
        "statistical_ranking_supported": False,
        "compatibility_is_in_band_proof": False,
        "next_round_l_values": result.next_round_l_values,
        "next_round_selection": "compatible_primaries_union_compatible_one_hop_coverage",
        "next_round_ranking_performed": False,
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
        "schema": "bayesfilter.pp_ukf.statistical_compatibility_guard_manifest.v1",
        "status": status,
        "started_utc": started_utc.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
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
        "target_signature": repair.TARGET_SIGNATURE,
        "adapter_signature": adapter_signature,
        "transport_path": args.frozen_transport,
        "transport_sha256": repair.TRANSPORT_SHA256,
        "metric_policy": "fixed_identity",
        "plan_path": PLAN,
        "source_manifest": SOURCE_MANIFEST,
        "prior_charged_seconds": float(source_manifest["cumulative_charged_seconds"]),
        "wall_seconds": wall,
        "cumulative_charged_seconds": (
            float(source_manifest["cumulative_charged_seconds"]) + wall
        ),
        "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
        "all_tuning_draws_discarded": True,
        "sampling_launched": False,
    }
    base._write_new_json(args.output_root / "run_manifest.json", manifest)
    base._write_progress_json(
        args.output_root / "progress.json",
        {
            **_progress_payload(
                started=started,
                reconstruction_receipts=reconstruction_receipts,
                guards=guards,
                planned_guard_count=len(guard_requests),
                guard_failures=guard_failures,
                terminal=True,
            ),
            "status": status,
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
