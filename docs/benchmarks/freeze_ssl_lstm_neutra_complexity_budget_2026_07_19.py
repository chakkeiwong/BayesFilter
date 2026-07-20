#!/usr/bin/env python3
"""Replay timing receipts and freeze the q-complexity ladder resource cap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "bayesfilter.ssl_lstm.neutra_hmc.complexity_budget_freeze.v1"
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-neutra-hmc-state-complexity-ladder-plan-2026-07-19.md"
)
RESULT = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-neutra-hmc-state-complexity-ladder-result-2026-07-19.md"
)
ARTIFACT_ROOT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-neutra-hmc-state-complexity-2026-07-19"
)
Q_VALUES = (1, 2, 5, 10, 20)
WORKERS_BY_Q = {1: 32, 2: 32, 5: 32, 10: 32, 20: 16}
HOST_RAM_CAP_BYTES = 64 * 1024**3
BUDGET_MARGIN = 1.50

OPTUNA_TRIALS = 6
OPTUNA_STREAMS = 2
OPTUNA_MAX_STEPS = 400
FINAL_STREAMS = 2
FINAL_MAX_STEPS = 5000
FRESH_CONFIRMATION_STREAMS = 1
FRESH_CONFIRMATION_MAX_STEPS = 5000
PHASE3_TOTAL_PROSPECTIVE_STEPS = 19_800
PHASE3_FRESH_POOL_LAUNCHES = 3
PHASE3_TRAINER_CONSTRUCTIONS = 15
PHASE3_TRAINERS_COVERED_BY_CANARY_LAUNCHES = 6
PHASE3_ADDITIONAL_TRAINER_COLD_STARTS = 9

HMC_TRANSITION_LEAPFROGS_PER_RUNG = 408_800
HMC_COLD_RESERVE_SECONDS_PER_RUNG = 9_000.0
FORECAST_TOTAL_BLOCKS = 388
FORECAST_FRESH_POOL_STARTS = 3
FORECAST_WARM_BLOCKS = 385

PHASE3_RATE_RECEIPTS = {
    1: "process-parallel/canary-q1-w32-principal-readiness.json",
    2: "process-parallel/canary-q2-w32-principal-readiness.json",
    5: "process-parallel/topology-q5-w32-principal-readiness.json",
    10: "process-parallel/topology-q10-w32-principal-readiness.json",
    20: "process-parallel/canary-q20-w16-principal-readiness.json",
}
PHASE3_STARTUP_RECEIPTS = {
    q: f"process-parallel/canary-q{q}-w{WORKERS_BY_Q[q] if q in (1, 2, 20) else 16}-principal-readiness.json"
    for q in Q_VALUES
}
PHASE3_STARTUP_WORKERS = {1: 32, 2: 32, 5: 16, 10: 16, 20: 16}

PHASE3_CURRENT_SOURCES = {
    "target": Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
    "pool": Path("bayesfilter/inference/cpu_value_score_pool.py"),
    "trainer": Path("bayesfilter/inference/neutra_training.py"),
}
HMC_EXECUTION_SOURCES = (
    Path("docs/benchmarks/benchmark_ssl_lstm_complexity_hmc_budget_rate_2026_07_19.py"),
    Path("bayesfilter/inference/hmc.py"),
    Path("bayesfilter/inference/batched_value_score.py"),
    Path("bayesfilter/inference/neutra_training.py"),
    Path("bayesfilter/inference/neutra_artifacts.py"),
    Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
)
FORECAST_EXECUTION_SOURCES = (
    Path("docs/benchmarks/benchmark_ssl_lstm_complexity_forecast_pool_2026_07_19.py"),
    Path("bayesfilter/inference/cpu_forecast_pool.py"),
    Path("bayesfilter/nonlinear/ssl_lstm_complexity_predictive_tf.py"),
    Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
)


class BudgetFreezeError(RuntimeError):
    pass


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BudgetFreezeError(f"expected JSON object: {path}")
    return value


def repo_path(path: Path, *, label: str) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise BudgetFreezeError(f"{label} must remain inside the repository")
    return resolved


def source_signature(paths: tuple[Path, ...]) -> str:
    return payload_sha256({path.as_posix(): sha256(ROOT / path) for path in paths})


def receipt_binding(relative: Path) -> dict[str, str]:
    path = repo_path(relative, label="receipt")
    if not path.is_file():
        raise BudgetFreezeError(f"missing receipt: {relative}")
    return {"path": relative.as_posix(), "sha256": sha256(path)}


def assert_close(actual: float, expected: float, *, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1.0e-12, abs_tol=1.0e-9):
        raise BudgetFreezeError(f"{label} mismatch: {actual} != {expected}")


def phase3_warm_rate(receipt: Mapping[str, Any]) -> float:
    if "warm_summary" in receipt:
        return float(receipt["warm_summary"]["wall_seconds_max"])
    streams = receipt.get("streams")
    if not isinstance(streams, list) or len(streams) != 2:
        raise BudgetFreezeError("Phase 3 rate receipt must contain two streams")
    return max(float(row["warm_step_max_seconds"]) for row in streams)


def validate_phase3_receipt(
    receipt: Mapping[str, Any],
    *,
    q: int,
    expected_sources: Mapping[str, str],
    expected_worker_count: int,
) -> None:
    if int(receipt.get("q", -1)) != q:
        raise BudgetFreezeError("Phase 3 receipt q mismatch")
    if receipt.get("hard_vetoes") not in (None, []):
        raise BudgetFreezeError("Phase 3 timing receipt contains a hard veto")
    manifest = receipt.get("run_manifest", {})
    source_hashes = receipt.get("source_hashes", manifest.get("source_hashes", {}))
    for key, expected in expected_sources.items():
        if source_hashes.get(key) != expected:
            raise BudgetFreezeError(f"Phase 3 {key} source drift at q={q}")
    worker_count = manifest.get("value_score_worker_count", receipt.get("workers"))
    if worker_count is not None and int(worker_count) != int(expected_worker_count):
        raise BudgetFreezeError(f"Phase 3 selected worker-count mismatch at q={q}")
    memory = receipt.get("memory", {})
    host_bytes = memory.get(
        "combined_conservative_bytes",
        manifest.get("combined_conservative_host_bytes"),
    )
    if host_bytes is None or int(host_bytes) > HOST_RAM_CAP_BYTES:
        raise BudgetFreezeError(f"Phase 3 host-memory evidence invalid at q={q}")
    if phase3_warm_rate(receipt) <= 0.0:
        raise BudgetFreezeError(f"Phase 3 warm rate is invalid at q={q}")


def phase3_startup_terms(receipt: Mapping[str, Any]) -> tuple[float, float]:
    streams = receipt.get("streams")
    if not isinstance(streams, list) or len(streams) != 2:
        raise BudgetFreezeError("Phase 3 startup receipt must contain two streams")
    wall = float(receipt["run_manifest"]["wall_seconds"])
    second = streams[1]
    additional_trainer_cold = max(
        0.0,
        float(second["first_step_seconds"]) - float(second["warm_step_max_seconds"]),
    )
    if wall <= 0.0:
        raise BudgetFreezeError("Phase 3 full-canary wall is invalid")
    return wall, additional_trainer_cold


def phase3_budget_row(q: int, expected_sources: Mapping[str, str]) -> dict[str, Any]:
    rate_relative = ARTIFACT_ROOT / PHASE3_RATE_RECEIPTS[q]
    startup_relative = ARTIFACT_ROOT / PHASE3_STARTUP_RECEIPTS[q]
    rate = strict_json(ROOT / rate_relative)
    startup = strict_json(ROOT / startup_relative)
    validate_phase3_receipt(
        rate,
        q=q,
        expected_sources=expected_sources,
        expected_worker_count=WORKERS_BY_Q[q],
    )
    validate_phase3_receipt(
        startup,
        q=q,
        expected_sources=expected_sources,
        expected_worker_count=PHASE3_STARTUP_WORKERS[q],
    )
    warm_seconds = phase3_warm_rate(rate) * PHASE3_TOTAL_PROSPECTIVE_STEPS
    full_canary_wall, additional_trainer_cold = phase3_startup_terms(startup)
    if "calls" in rate:
        selected_first_call = float(rate["calls"][0]["wall_seconds"])
        if full_canary_wall < selected_first_call:
            raise BudgetFreezeError(
                f"Phase 3 startup reserve does not dominate selected first call at q={q}"
            )
    cold_seconds = (
        PHASE3_FRESH_POOL_LAUNCHES * full_canary_wall
        + PHASE3_ADDITIONAL_TRAINER_COLD_STARTS * additional_trainer_cold
    )
    budget_seconds = BUDGET_MARGIN * (warm_seconds + cold_seconds)
    return {
        "q": q,
        "selected_worker_count": WORKERS_BY_Q[q],
        "startup_receipt_worker_count": PHASE3_STARTUP_WORKERS[q],
        "rate_receipt": receipt_binding(rate_relative),
        "startup_receipt": receipt_binding(startup_relative),
        "warm_seconds_per_training_step": phase3_warm_rate(rate),
        "prospective_training_steps": PHASE3_TOTAL_PROSPECTIVE_STEPS,
        "warm_work_seconds_before_margin": warm_seconds,
        "full_canary_startup_wall_seconds": full_canary_wall,
        "additional_trainer_cold_seconds": additional_trainer_cold,
        "fresh_pool_launches": PHASE3_FRESH_POOL_LAUNCHES,
        "additional_trainer_cold_starts": PHASE3_ADDITIONAL_TRAINER_COLD_STARTS,
        "cold_reserve_seconds_before_margin": cold_seconds,
        "margin": BUDGET_MARGIN,
        "budget_seconds": budget_seconds,
        "budget_hours": budget_seconds / 3600.0,
    }


def hmc_budget_row(q: int, expected_signature: str) -> dict[str, Any]:
    relative = ARTIFACT_ROOT / "hmc-budget-rate" / f"hmc-budget-rate-q{q}.json"
    receipt = strict_json(ROOT / relative)
    if (
        receipt.get("schema") != "bayesfilter.ssl_lstm.complexity_hmc_budget_rate.v1"
        or receipt.get("status") != "PASSED"
        or receipt.get("mode") != "timing-canary"
        or int(receipt.get("q", -1)) != q
    ):
        raise BudgetFreezeError(f"invalid HMC receipt at q={q}")
    if receipt.get("execution_source_signature") != expected_signature:
        raise BudgetFreezeError(f"HMC execution-source drift at q={q}")
    if receipt.get("selected_hmc_topology") != "single_tfp_sample_chain_batched_four_chain_xla":
        raise BudgetFreezeError(f"HMC topology mismatch at q={q}")
    if int(receipt.get("hmc_transition_leapfrogs_per_rung", -1)) != HMC_TRANSITION_LEAPFROGS_PER_RUNG:
        raise BudgetFreezeError(f"HMC operation-count mismatch at q={q}")
    assert_close(receipt["hmc_margin"], BUDGET_MARGIN, label="HMC margin")
    assert_close(
        receipt["hmc_cold_reserve_seconds_per_rung"],
        HMC_COLD_RESERVE_SECONDS_PER_RUNG,
        label="HMC cold reserve",
    )
    rate = float(receipt["warm_seconds_per_transition_leapfrog_max"])
    expected_budget = (
        BUDGET_MARGIN * rate * HMC_TRANSITION_LEAPFROGS_PER_RUNG
        + HMC_COLD_RESERVE_SECONDS_PER_RUNG
    )
    assert_close(receipt["hmc_budget_seconds"], expected_budget, label="HMC budget")
    if int(receipt["run_manifest"]["host_ru_maxrss_bytes"]) > HOST_RAM_CAP_BYTES:
        raise BudgetFreezeError(f"HMC host-memory evidence invalid at q={q}")
    return {
        "q": q,
        "receipt": receipt_binding(relative),
        "execution_source_signature": expected_signature,
        "warm_seconds_per_transition_leapfrog": rate,
        "transition_leapfrogs": HMC_TRANSITION_LEAPFROGS_PER_RUNG,
        "cold_reserve_seconds": HMC_COLD_RESERVE_SECONDS_PER_RUNG,
        "margin": BUDGET_MARGIN,
        "budget_seconds": expected_budget,
        "budget_hours": expected_budget / 3600.0,
    }


def forecast_budget_row(q: int, expected_signature: str) -> dict[str, Any]:
    relative = (
        ARTIFACT_ROOT
        / "forecast-timing"
        / f"forecast-timing-q{q}-repair-01.json"
    )
    receipt = strict_json(ROOT / relative)
    if (
        receipt.get("schema") != "bayesfilter.ssl_lstm.complexity_forecast_pool_timing.v1"
        or receipt.get("status") != "PASSED"
        or receipt.get("mode") != "timing-canary"
        or int(receipt.get("q", -1)) != q
    ):
        raise BudgetFreezeError(f"invalid forecast receipt at q={q}")
    if receipt.get("execution_source_signature") != expected_signature:
        raise BudgetFreezeError(f"forecast execution-source drift at q={q}")
    if int(receipt.get("worker_count", -1)) != WORKERS_BY_Q[q]:
        raise BudgetFreezeError(f"forecast worker-count mismatch at q={q}")
    if (
        int(receipt.get("block_draws", -1)) != 256
        or int(receipt.get("forecast_replication_count", -1)) != 2
        or int(receipt.get("forecast_horizon", -1)) != 10
        or receipt.get("seed_domain_disjoint_from_material") is not True
    ):
        raise BudgetFreezeError(f"forecast workload contract mismatch at q={q}")
    first = float(receipt["first_block_seconds"])
    warm = float(receipt["warm_block_max_seconds"])
    expected_budget = BUDGET_MARGIN * (
        FORECAST_FRESH_POOL_STARTS * first + FORECAST_WARM_BLOCKS * warm
    )
    assert_close(
        receipt["phase6_forecast_budget_seconds_with_50pct_margin"],
        expected_budget,
        label="forecast budget",
    )
    calls = receipt.get("calls")
    if not isinstance(calls, list) or len(calls) != 3:
        raise BudgetFreezeError(f"forecast replay count mismatch at q={q}")
    for call in calls:
        worker = call.get("worker_metadata", {})
        if int(worker.get("aggregate_parent_worker_ru_maxrss_bytes", HOST_RAM_CAP_BYTES + 1)) > HOST_RAM_CAP_BYTES:
            raise BudgetFreezeError(f"forecast host-memory evidence invalid at q={q}")
        if int(worker.get("configured_worker_count", -1)) != WORKERS_BY_Q[q]:
            raise BudgetFreezeError(f"forecast configured workers mismatch at q={q}")
        if int(worker.get("active_worker_count", -1)) != WORKERS_BY_Q[q]:
            raise BudgetFreezeError(f"forecast active workers mismatch at q={q}")
    return {
        "q": q,
        "receipt": receipt_binding(relative),
        "execution_source_signature": expected_signature,
        "first_block_seconds": first,
        "warm_block_max_seconds": warm,
        "total_blocks": FORECAST_TOTAL_BLOCKS,
        "fresh_pool_starts": FORECAST_FRESH_POOL_STARTS,
        "warm_blocks": FORECAST_WARM_BLOCKS,
        "margin": BUDGET_MARGIN,
        "budget_seconds": expected_budget,
        "budget_hours": expected_budget / 3600.0,
    }


def build_budget() -> dict[str, Any]:
    phase3_source_hashes = {
        key: sha256(ROOT / path) for key, path in PHASE3_CURRENT_SOURCES.items()
    }
    hmc_signature = source_signature(HMC_EXECUTION_SOURCES)
    forecast_signature = source_signature(FORECAST_EXECUTION_SOURCES)
    phase3_rows = [phase3_budget_row(q, phase3_source_hashes) for q in Q_VALUES]
    hmc_rows = [hmc_budget_row(q, hmc_signature) for q in Q_VALUES]
    forecast_rows = [forecast_budget_row(q, forecast_signature) for q in Q_VALUES]
    phase3_seconds = sum(row["budget_seconds"] for row in phase3_rows)
    hmc_seconds = sum(row["budget_seconds"] for row in hmc_rows)
    forecast_seconds = sum(row["budget_seconds"] for row in forecast_rows)
    gpu_seconds = phase3_seconds + hmc_seconds
    total_seconds = gpu_seconds + forecast_seconds
    current_sources = {
        path.as_posix(): sha256(ROOT / path)
        for path in (
            SCRIPT,
            Path("docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py"),
            Path("bayesfilter/inference/neutra_training_control.py"),
            *PHASE3_CURRENT_SOURCES.values(),
            *HMC_EXECUTION_SOURCES,
            *FORECAST_EXECUTION_SOURCES,
        )
    }
    payload = {
        "schema": SCHEMA,
        "status": "BUDGET_FROZEN_MATERIAL_LAUNCH_UNAUTHORIZED",
        "q_order": list(Q_VALUES),
        "research_question": (
            "What conservative sequential wall/GPU/CPU cap covers the implemented "
            "Phase 3-6 q-complexity ladder before scientific execution?"
        ),
        "evidence_contract": {
            "exact_baseline": "current q-general Phase 3-6 runners and selected topologies",
            "primary_criterion": "all bound timing receipts replay exactly under current execution sources",
            "hard_vetoes": [
                "source or receipt hash drift",
                "schema, q, topology, workload, seed-domain, or operation-count mismatch",
                "receipt hard veto or host-memory evidence above 64 GiB",
                "budget arithmetic mismatch",
            ],
            "explanatory_only": [
                "continuous timing differences by q",
                "acceptance, movement, loss, and per-worker timing rows in source receipts",
            ],
            "nonclaims": [
                "no NeuTra quality, HMC convergence, posterior correctness, predictive validity, or superiority claim",
                "the cap is a conservative authorization ceiling, not an expected runtime",
                "budget freeze does not authorize material execution",
            ],
        },
        "phase3_contract": {
            "optuna_steps": OPTUNA_TRIALS * OPTUNA_STREAMS * OPTUNA_MAX_STEPS,
            "final_steps": FINAL_STREAMS * FINAL_MAX_STEPS,
            "fresh_confirmation_steps": (
                FRESH_CONFIRMATION_STREAMS * FRESH_CONFIRMATION_MAX_STEPS
            ),
            "total_prospective_steps_per_q": PHASE3_TOTAL_PROSPECTIVE_STEPS,
            "fresh_pool_launches_per_q": PHASE3_FRESH_POOL_LAUNCHES,
            "trainer_constructions_per_q": PHASE3_TRAINER_CONSTRUCTIONS,
            "trainer_constructions_covered_by_three_full_canaries": (
                PHASE3_TRAINERS_COVERED_BY_CANARY_LAUNCHES
            ),
            "additional_trainer_cold_starts_per_q": (
                PHASE3_ADDITIONAL_TRAINER_COLD_STARTS
            ),
            "startup_reserve_rule": (
                "1.5 * (3 * full two-stream canary wall + 9 * max(0, "
                "second-stream first-step minus second-stream warm max))"
            ),
            "source_sha256": phase3_source_hashes,
            "rows": phase3_rows,
            "subtotal_seconds": phase3_seconds,
            "subtotal_hours": phase3_seconds / 3600.0,
        },
        "hmc_contract": {
            "execution_source_signature": hmc_signature,
            "rows": hmc_rows,
            "subtotal_seconds": hmc_seconds,
            "subtotal_hours": hmc_seconds / 3600.0,
        },
        "forecast_contract": {
            "execution_source_signature": forecast_signature,
            "rows": forecast_rows,
            "subtotal_seconds": forecast_seconds,
            "subtotal_hours": forecast_seconds / 3600.0,
        },
        "totals": {
            "gpu_active_seconds": gpu_seconds,
            "gpu_active_hours": gpu_seconds / 3600.0,
            "cpu_only_forecast_seconds": forecast_seconds,
            "cpu_only_forecast_hours": forecast_seconds / 3600.0,
            "sequential_wall_cap_seconds": total_seconds,
            "sequential_wall_cap_hours": total_seconds / 3600.0,
        },
        "sequential_stopping": {
            "enabled": True,
            "returns_unused_budget": True,
            "may_exceed_frozen_cap": False,
            "material_launch_authorized": False,
        },
        "source_bindings": current_sources,
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(
                    ("git", "status", "--porcelain"), cwd=ROOT, text=True
                ).strip()
            ),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "cpu_status": "CPU-only receipt parsing and SHA-256 replay",
            "device_use": "none_cpu_only_receipt_replay",
            "gpu_status": "not_initialized_or_used",
            "jit_compile": "N/A",
            "tf32": "N/A",
            "random_seeds": "N/A_receipt_replay_only",
            "data_version": "bound by target/transport signatures in source receipts",
            "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
            "plan": PLAN.as_posix(),
            "result": RESULT.as_posix(),
        },
        "decision_table": {
            "decision": "freeze cap and request separate material launch authority",
            "primary_criterion_status": "passed",
            "veto_diagnostic_status": "passed",
            "main_uncertainty": (
                "short timing canaries extrapolate long stochastic runs; the 50% margins, "
                "explicit cold reserves, and sequential stopping are conservative controls"
            ),
            "next_justified_action": "request explicit material Phase 3-6 launch authority",
            "not_concluded": (
                "transport quality, convergence, posterior correctness, predictive validity, "
                "scientific superiority, or expected runtime"
            ),
        },
        "inference_status": {
            "hard_veto_screen": "passed for receipt identity, mechanics, and resources",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "all per-q timing differences",
            "default_readiness": "not assessed",
            "next_evidence_needed": "material sequential ladder under separate authority",
        },
        "post_run_red_team": {
            "strongest_alternative_explanation": (
                "shared-host load or longer-run cache behavior may make realized wall time differ "
                "from the short canaries"
            ),
            "result_that_would_overturn": (
                "source/receipt drift, invalid operation counts, or a material run breaching a "
                "stated resource/validity veto"
            ),
            "weakest_evidence": "warm-rate extrapolation to the q=20 5,000-step and L=16 envelopes",
        },
    }
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise BudgetFreezeError(f"refusing to overwrite budget artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = repo_path(args.output, label="output")
    started = time.perf_counter()
    payload = build_budget()
    payload["run_manifest"]["wall_seconds"] = time.perf_counter() - started
    payload["run_manifest"]["output"] = args.output.as_posix()
    payload["payload_signature_excluding_this_field"] = payload_sha256(payload)
    write_json(output, payload)
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {
                "status": payload["status"],
                "gpu_active_hours": payload["totals"]["gpu_active_hours"],
                "cpu_only_forecast_hours": payload["totals"][
                    "cpu_only_forecast_hours"
                ],
                "sequential_wall_cap_hours": payload["totals"][
                    "sequential_wall_cap_hours"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
