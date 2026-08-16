#!/usr/bin/env python3
"""Bounded recovery for the q=20 GPU/XLA budget preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_direct_gpu_xla_r2_budget_preflight_2026_07_30.py"
)
SPEC = importlib.util.spec_from_file_location("q20_r2_preflight_base", BASE_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load base q=20 r2 preflight runner")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


SCHEMA = "bayesfilter.ssl_lstm.q20_direct_gpu_xla_r2_budget_preflight_recovery.v1"
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-recovery-plan-2026-07-30.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
DEFAULT_OUTPUT_ROOT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-2026-07-30/r2"
)
PRIOR_OUTPUT_ROOT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-2026-07-30/r1"
)
PRIOR_PROGRESS = PRIOR_OUTPUT_ROOT / "timing/32x32/progress.json"
PRIOR_LEDGER = PRIOR_OUTPUT_ROOT / "material-budget-ledger.json"
MATERIAL_CAP_SECONDS = 6992.0
PRIOR_MATERIAL_SECONDS = 5007.6122855830035
AUTHORIZED_REMAINING_SECONDS = 8177.078434643995
RESERVED_NONMATERIAL_SECONDS = AUTHORIZED_REMAINING_SECONDS - MATERIAL_CAP_SECONDS
WARM_UPDATE_COUNT = 3

SOURCE_PATHS = {
    **base.SOURCE_PATHS,
    "base_runner": BASE_RUNNER.relative_to(ROOT),
    "recovery_runner": SCRIPT,
    "plan": PLAN,
}
SOURCE_SHA256 = {
    key: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    for key, path in SOURCE_PATHS.items()
}


def live_source_sha256() -> Mapping[str, str]:
    return {
        key: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for key, path in SOURCE_PATHS.items()
    }

# Reuse the reviewed primitives while rebinding every artifact to this revision.
base.SCHEMA = SCHEMA
base.PLAN = PLAN
base.SCRIPT = SCRIPT
base.SOURCE_PATHS = SOURCE_PATHS
base.SOURCE_SHA256 = SOURCE_SHA256
base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
base.MATERIAL_CAP_SECONDS = MATERIAL_CAP_SECONDS
base.AUTHORIZED_REMAINING_SECONDS = AUTHORIZED_REMAINING_SECONDS
base.RESERVED_NONMATERIAL_SECONDS = RESERVED_NONMATERIAL_SECONDS


def _prior_progress() -> Mapping[str, Any]:
    payload = base.read_json(ROOT / PRIOR_PROGRESS)
    operations = {str(row["name"]): row for row in payload.get("operations", ())}
    required = {
        "target_and_binding_construction",
        "trainer_construction",
        "validation_64_first",
        "validation_64_warm",
        "optimizer_update_1",
        "optimizer_update_2",
        "optimizer_update_3",
        "optimizer_update_4",
        "optimizer_update_5",
        "optimizer_update_6",
        "status_probe_2",
        "support_export_first",
        "support_export_warm",
    }
    missing = sorted(required.difference(operations))
    if missing:
        raise base.PreflightError(f"preserved 32x32 receipts are incomplete: {missing}")
    return payload


def exact_shape_validation_result(
    trainer: Any,
    target: Any,
    z: Any,
    *,
    batch_size: int,
) -> Mapping[str, Any]:
    spec = base.tf.TensorSpec([int(batch_size), 4], base.tf.float64)
    compiled = base.tf.function(
        trainer._validation_impl,
        input_signature=[spec],
        jit_compile=True,
        reduce_retracing=False,
    )
    rows = compiled(base.tf.ensure_shape(z, [int(batch_size), 4]))
    per_sample_loss = rows[0]
    theta = rows[2]
    if not base.tensor_finite(per_sample_loss):
        raise base.PreflightError("exact-shape validation loss is nonfinite")
    _value, _score, status = target.batch_value_score_status(theta)
    hard_valid = bool(
        base.tf.reduce_all(status["hard_valid_for_training"]).numpy()
    )
    if not hard_valid:
        raise base.PreflightError("exact-shape validation target status failed")
    return {
        "row_count": int(batch_size),
        "static_input_shape": spec.shape.as_list(),
        "mean_loss": float(base.tf.reduce_mean(per_sample_loss).numpy()),
        "all_finite": True,
        "all_target_status_valid": hard_valid,
        "floor_count_total": int(
            base.tf.reduce_sum(status["floor_count_value"]).numpy()
        ),
        "min_innovation_eigenvalue": float(
            base.tf.reduce_min(status["min_innovation_eigenvalue"]).numpy()
        ),
    }


def run_recovery_timing(
    args: argparse.Namespace,
    budget: Any,
) -> Mapping[str, Any]:
    identity = base.read_json(ROOT / args.output_root / "identity-comparison.json")
    if identity.get("status") != "TARGET_IDENTITY_PARITY_PASSED":
        raise base.PreflightError("recovery timing requires passing identity parity")
    prior = _prior_progress()
    arm_root = ROOT / args.output_root / "timing/64x64"
    progress = base.ProgressRecorder(
        arm_root / "progress.json",
        {
            "architecture": [64, 64],
            "batch_size": base.BATCH_SIZE,
            "learning_rate": 2.0e-4,
            "source_sha256": SOURCE_SHA256,
            "identity_comparison_sha256": base.sha256(
                ROOT / args.output_root / "identity-comparison.json"
            ),
            "preserved_32x32_progress_path": PRIOR_PROGRESS.as_posix(),
            "preserved_32x32_progress_sha256": base.sha256(ROOT / PRIOR_PROGRESS),
            "preserved_r1_ledger_path": PRIOR_LEDGER.as_posix(),
            "preserved_r1_ledger_sha256": base.sha256(ROOT / PRIOR_LEDGER),
            "prior_material_seconds": PRIOR_MATERIAL_SECONDS,
            "run_manifest": base.run_manifest(args),
        },
        budget,
    )
    state: dict[str, Any] = {}

    def construct_target() -> Mapping[str, Any]:
        owner, binding, target = base.bound_target()
        if owner.target_signature() != identity.get("target_signature"):
            raise base.PreflightError("timing target signature differs from identity gate")
        if SOURCE_SHA256 != live_source_sha256():
            raise base.PreflightError("live source closure changed after identity issuance")
        state.update(owner=owner, binding=binding, target=target)
        return {
            "target_signature": owner.target_signature(),
            "adapter_signature": owner.adapter_signature(),
            "binding": binding.payload(),
            "preserved_32x32_operation_count": len(prior["operations"]),
        }

    progress.run("target_and_binding_construction", construct_target)

    def construct_trainer() -> Mapping[str, Any]:
        trainer = base.make_trainer(state["target"], (64, 64))
        state["trainer"] = trainer
        return {
            "variable_devices": sorted(
                {variable.device for variable in trainer.variables}
            ),
            "trainable_variable_count": len(trainer.variables),
            "trainer_state_hash": trainer.state_payload()["state_hash"],
        }

    progress.run("trainer_construction", construct_trainer)
    trainer = state["trainer"]
    target = state["target"]

    for step_index in range(1, WARM_UPDATE_COUNT + 2):
        z = base.stateless_batch((20260730, 9401), step_index, base.BATCH_SIZE)
        progress.run(
            f"optimizer_update_{step_index}",
            lambda z=z: base.step_payload(trainer.train_step(z)),
        )

    validation_z = base.stateless_batch(
        (20260730, 9201), 0, base.VALIDATION_SIZE
    )
    progress.run(
        "validation_64_compile_inclusive",
        lambda: exact_shape_validation_result(
            trainer,
            target,
            validation_z,
            batch_size=base.VALIDATION_SIZE,
        ),
    )

    status_z = base.stateless_batch((20260730, 9501), 0, 2)

    def status_probe() -> Mapping[str, Any]:
        theta, _logdet = trainer.forward_and_logdet(status_z)
        _value, _score, status = target.batch_value_score_status(theta)
        valid = bool(base.tf.reduce_all(status["hard_valid_for_training"]).numpy())
        if not valid:
            raise base.PreflightError("two-row status probe failed")
        return {
            "row_count": 2,
            "all_hard_valid": valid,
            "floor_count_total": int(
                base.tf.reduce_sum(status["floor_count_value"]).numpy()
            ),
        }

    progress.run("status_probe_2", status_probe)
    progress.run(
        "support_export_single",
        lambda: base.support_result(trainer, target, "recovery-64x64"),
    )

    audit_z = base.stateless_batch((20260730, 9301), 0, base.AUDIT_SIZE)
    progress.run(
        "audit_shape_256_compile_inclusive",
        lambda: exact_shape_validation_result(
            trainer,
            target,
            audit_z,
            batch_size=base.AUDIT_SIZE,
        ),
    )

    def hlo_receipt() -> Mapping[str, Any]:
        z = base.stateless_batch((20260730, 9601), 0, base.BATCH_SIZE)
        hlo = trainer._compiled_train_step.experimental_get_compiler_ir(z)(stage="hlo")
        encoded = hlo if isinstance(hlo, bytes) else str(hlo).encode("utf-8")
        return {"hlo_sha256": hashlib.sha256(encoded).hexdigest()}

    progress.run("hlo_extraction", hlo_receipt)
    complete = progress.complete()
    result = {
        **complete,
        "status": "RECOVERY_TIMING_DIAGNOSTIC_COMPLETED",
        "progress_path": progress.path.relative_to(ROOT).as_posix(),
        "progress_sha256": base.sha256(progress.path),
        "nonclaims": [
            "timing and mechanics only",
            "no tuning selection, candidate rejection, HMC, posterior, or default claim",
        ],
    }
    base.write_json(arm_root / "result.json", result)
    return result


def _duration_rows(payload: Mapping[str, Any]) -> Mapping[str, float]:
    return {
        str(row["name"]): float(row["duration_seconds"])
        for row in payload.get("operations", ())
    }


def _architecture_summaries(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Mapping[str, Mapping[str, Any]]:
    prior = _prior_progress()
    current = base.read_json(ROOT / output_root / "timing/64x64/result.json")
    old = _duration_rows(prior)
    new = _duration_rows(current)
    audit = new["audit_shape_256_compile_inclusive"]
    summary32 = {
        "construction_seconds": old["target_and_binding_construction"]
        + old["trainer_construction"],
        "validation_64_first_seconds": old["validation_64_first"],
        "validation_64_warm_seconds": old["validation_64_warm"],
        "optimizer_update_first_seconds": old["optimizer_update_1"],
        "optimizer_update_warm_seconds": [old[f"optimizer_update_{i}"] for i in range(2, 7)],
        "optimizer_update_warm_median_seconds": base.statistics.median(
            old[f"optimizer_update_{i}"] for i in range(2, 7)
        ),
        "optimizer_update_warm_max_seconds": max(
            old[f"optimizer_update_{i}"] for i in range(2, 7)
        ),
        "optimizer_update_warm_min_seconds": min(
            old[f"optimizer_update_{i}"] for i in range(2, 7)
        ),
        "status_probe_2_seconds": old["status_probe_2"],
        "support_export_first_seconds": old["support_export_first"],
        "support_export_warm_seconds": old["support_export_warm"],
        "audit_shape_256_first_seconds": audit,
        "audit_shape_256_warm_seconds": audit,
        "hlo_extraction_seconds": new["hlo_extraction"],
        "audit_estimate_provenance": "direct 64x64 compile-inclusive upper planning substitution",
        "hlo_estimate_provenance": "direct 64x64 shared upper planning substitution",
    }
    warm64 = [new[f"optimizer_update_{i}"] for i in range(2, 5)]
    validation64 = new["validation_64_compile_inclusive"]
    support64 = new["support_export_single"]
    summary64 = {
        "construction_seconds": new["target_and_binding_construction"]
        + new["trainer_construction"],
        "validation_64_first_seconds": validation64,
        "validation_64_warm_seconds": validation64,
        "optimizer_update_first_seconds": new["optimizer_update_1"],
        "optimizer_update_warm_seconds": warm64,
        "optimizer_update_warm_median_seconds": base.statistics.median(warm64),
        "optimizer_update_warm_max_seconds": max(warm64),
        "optimizer_update_warm_min_seconds": min(warm64),
        "status_probe_2_seconds": new["status_probe_2"],
        "support_export_first_seconds": support64,
        "support_export_warm_seconds": support64,
        "audit_shape_256_first_seconds": audit,
        "audit_shape_256_warm_seconds": audit,
        "hlo_extraction_seconds": new["hlo_extraction"],
        "validation_estimate_provenance": "direct compile-inclusive cost used for first and warm calls",
        "support_estimate_provenance": "direct single-call cost used for first and warm calls",
        "audit_estimate_provenance": "direct compile-inclusive cost used for first and warm calls",
    }
    return {"32x32": summary32, "64x64": summary64}


def project_budget(args: argparse.Namespace) -> Mapping[str, Any]:
    identity = base.read_json(ROOT / args.output_root / "identity-comparison.json")
    if identity.get("status") != "TARGET_IDENTITY_PARITY_PASSED":
        raise base.PreflightError("projection requires passing identity parity")
    summaries = _architecture_summaries(args.output_root)

    def scenario(selected_final: str, use_max: bool) -> float:
        tuning = sum(
            base.projected_process_cost(
                summaries[label],
                updates=100,
                validation_calls=3,
                support_calls=1,
                audit_calls=0,
                use_max=use_max,
            )
            * 2.0
            for label in ("32x32", "64x64")
        )
        finals = base.projected_process_cost(
            summaries[selected_final],
            updates=1000,
            validation_calls=11,
            support_calls=12,
            audit_calls=2,
            use_max=use_max,
        ) * 2.0
        return tuning + finals

    scenarios = {}
    for selected in ("32x32", "64x64"):
        median_total = scenario(selected, False)
        max_total = scenario(selected, True)
        scenarios[selected] = {
            "unbuffered_warm_median_seconds": median_total,
            "unbuffered_warm_max_seconds": max_total,
            "buffered_warm_median_seconds": median_total * base.CONTINGENCY_FACTOR,
            "buffered_warm_max_seconds": max_total * base.CONTINGENCY_FACTOR,
        }
    requested = max(row["buffered_warm_max_seconds"] for row in scenarios.values())
    payload = {
        "schema": f"{SCHEMA}.projection",
        "status": "BUDGET_PREFLIGHT_COMPLETED",
        "identity": identity,
        "timing": summaries,
        "projection_protocol": {
            "tuning_arms": 4,
            "tuning_updates_per_arm": 100,
            "tuning_validation_calls_per_arm": 3,
            "final_streams": 2,
            "final_updates_per_stream": 1000,
            "final_validation_calls_per_stream": 11,
            "final_support_calls_per_stream": 12,
            "final_audit_calls_per_stream": 2,
            "contingency_factor": base.CONTINGENCY_FACTOR,
        },
        "scenarios_by_selected_final_architecture": scenarios,
        "conservative_requested_campaign_seconds": requested,
        "conservative_requested_campaign_hours": requested / 3600.0,
        "conservative_requested_campaign_days": requested / 86400.0,
        "preserved_32x32_progress_path": PRIOR_PROGRESS.as_posix(),
        "preserved_32x32_progress_sha256": base.sha256(ROOT / PRIOR_PROGRESS),
        "recovery_64x64_result_path": (
            args.output_root / "timing/64x64/result.json"
        ).as_posix(),
        "recovery_64x64_result_sha256": base.sha256(
            ROOT / args.output_root / "timing/64x64/result.json"
        ),
        "numeric_provenance": {
            "optimizer_timing": "direct architecture-specific GPU/XLA receipts",
            "validation_support_audit_substitutions": "explicitly labeled conservative planning estimates",
            "protocol_counts": "inherited r1 full protocol for pricing only",
            "contingency_factor": "convenience planning margin, not an uncertainty interval",
        },
        "nonclaims": [
            "compute estimate only; no campaign authorization",
            "no tuning selection, training quality, convergence, HMC, posterior, or default claim",
        ],
    }
    base.write_json(ROOT / args.output_root / "projection.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("cpu-identity", "gpu-identity", "compare-identity", "timing", "project"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.architecture = "64x64" if args.mode == "timing" else None
    if args.output_root.is_absolute():
        raise base.PreflightError("output root must be repository-relative")
    output = (ROOT / args.output_root).resolve()
    if not output.is_relative_to(ROOT):
        raise base.PreflightError("output root escapes repository")
    budget = base.MaterialBudget(args.output_root) if args.mode in base.GPU_MODES else None
    try:
        if args.mode in {"cpu-identity", "gpu-identity"}:
            result = base.issue_identity(args)
        elif args.mode == "compare-identity":
            result = base.compare_identity(args)
        elif args.mode == "timing":
            assert budget is not None
            result = run_recovery_timing(args, budget)
        else:
            result = project_budget(args)
    except Exception:
        if budget is not None:
            budget.persist("FAILED_ATTEMPT")
        raise
    if budget is not None:
        budget.persist(str(result["status"]))
    print(json.dumps({"mode": args.mode, "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
