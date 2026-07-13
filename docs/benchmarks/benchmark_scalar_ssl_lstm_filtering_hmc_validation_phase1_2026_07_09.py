"""Phase 1 scalar SSL-LSTM filtering HMC validation screen.

This harness reruns the established scalar fixed-kernel replicated diagnostic
with fresh Phase 1 artifact paths and an explicit telemetry policy: unavailable
native divergence is not zero divergence.  The phase gate is a finite/acceptance
short-chain screen only.  It does not claim posterior correctness, convergence,
HMC readiness, zero divergences when native divergence is unavailable, GPU/XLA
readiness, default readiness, or sampler superiority.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCRIPT_NAME = "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py"
SCHEMA_VERSION = "scalar_ssl_lstm.filtering_hmc_validation_phase1.v1"
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md"
)
SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase1-cpu-short-chain-subplan-2026-07-09.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase1-cpu-short-chain-result-2026-07-09.md"
)
DEFAULT_GEOMETRY_PATH = (
    ROOT / "docs/benchmarks/scalar_ssl_lstm_filtering_geometry_cpu_hidden_2026-07-08.json"
)
DEFAULT_MASS_PATH = (
    ROOT / "docs/benchmarks/scalar_ssl_lstm_filtering_mass_handoff_cpu_hidden_2026-07-08.json"
)
DEFAULT_PHASE4_PATH = (
    ROOT / "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_short_smoke_cpu_hidden_2026-07-08.json"
)
DEFAULT_JSON_PATH = (
    ROOT / "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.json"
)
DEFAULT_MARKDOWN_PATH = (
    ROOT / "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.md"
)
BASE_REPLICATED_MODULE_PATH = (
    ROOT / "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_replicated_diagnostic_2026_07_08.py"
)
NONCLAIMS = (
    "Phase 1 finite/acceptance short-chain validation screen only",
    "not HMC readiness evidence",
    "not HMC convergence evidence",
    "not posterior correctness evidence",
    "not a zero-divergence claim when native divergence is unavailable",
    "not a tuned-kernel claim",
    "not sampler superiority evidence",
    "not statistically supported ranking evidence",
    "not GPU/XLA production-readiness evidence",
    "not default-readiness evidence",
    "not Zhao-Cui source-faithfulness evidence",
)


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_replicated_diagnostic_phase1_reuse",
        BASE_REPLICATED_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scalar replicated diagnostic module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def phase1_settings(base_module: Any) -> Any:
    return base_module.ReplicatedDiagnosticSettings(
        num_leapfrog_steps=4,
        step_size=0.3925,
        num_results=16,
        num_burnin_steps=4,
        seeds=(
            (20260709, 6101),
            (20260709, 6102),
            (20260709, 6103),
        ),
    )


def load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_phase1_validation(
    geometry_payload: Mapping[str, Any],
    mass_payload: Mapping[str, Any],
    phase4_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    start = time.perf_counter()
    base = load_base_module()
    settings = phase1_settings(base)
    base_payload = dict(
        base.run_replicated_diagnostic(
            geometry_payload,
            mass_payload,
            phase4_payload,
            settings=settings,
        )
    )
    gate = evaluate_phase1_gate(base_payload.get("seed_rows", ()), expected_seed_count=len(settings.seeds))
    base_decision = dict(base_payload.get("decision", {}))
    base_vetoes = tuple(base_decision.get("vetoes", ()))
    phase1_vetoes = tuple(dict.fromkeys((*base_vetoes, *gate["vetoes"])))
    passed = bool(not phase1_vetoes and base_decision.get("replicated_diagnostic_passed") is True)

    base_payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_role": "cpu_hidden_scalar_filtering_hmc_validation_phase1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "script": f"docs/benchmarks/{SCRIPT_NAME}",
            "plan_path": PLAN_PATH,
            "subplan_path": SUBPLAN_PATH,
            "result_path": RESULT_PATH,
            "settings": {
                **dict(base_payload.get("settings", {})),
                "phase1_gate_policy": phase1_gate_policy(),
            },
            "phase1_gate": gate,
            "telemetry_policy": telemetry_policy_payload(gate),
            "decision": {
                "phase1_short_chain_screen_passed": passed,
                "vetoes": phase1_vetoes,
                "base_replicated_diagnostic_passed": base_decision.get(
                    "replicated_diagnostic_passed"
                ),
                "base_replicated_vetoes": base_vetoes,
                "passed_seed_count": gate["passed_seed_count"],
                "seed_count": gate["seed_count"],
                "zero_divergence_claim_made": False,
                "viable_for_phase2_reference_agreement_subplan": passed,
                "next_justified_action": (
                    "write Phase 1 result and refresh/review Phase 2 reference-agreement subplan"
                    if passed
                    else "write Phase 1 blocker/repair result before Phase 2"
                ),
            },
            "metric_roles": {
                "phase1_short_chain_screen_passed": "primary_phase1_pass_fail",
                "finite_retained_samples": "hard_veto_evidence",
                "finite_target_log_prob": "hard_veto_evidence",
                "finite_log_accept_ratio": "hard_veto_evidence",
                "acceptance_strictly_between_0_05_and_0_99": "promotion_veto_screen",
                "native_divergence_positive_when_available": "hard_veto_evidence",
                "native_divergence_unavailable": "telemetry_availability_not_zero_divergences",
                "acceptance_rate_values": "descriptive_only_after_screen",
                "log_accept_tail_values": "descriptive_only_after_finiteness",
                "target_log_prob_range": "descriptive_only",
                "sample_range": "descriptive_only",
                "runtime": "explanatory_only",
            },
            "inference_status": {
                "hard_veto_screen": "passed" if passed else "failed",
                "native_divergence": gate["native_divergence_interpretation"],
                "zero_divergence_claim": "not made",
                "statistically_supported_ranking": (
                    "none; no method comparison and no uncertainty interval"
                ),
                "descriptive_only_differences": (
                    "per-seed acceptance, target-log-prob range, log-accept range, "
                    "sample range, and runtime"
                ),
                "default_readiness": "not assessed",
                "gpu_xla_readiness": "not assessed; CPU-hidden debug/reference exception",
                "hmc_readiness": "not assessed; Phase 1 finite/acceptance screen only",
                "next_evidence_needed": (
                    "reviewed Phase 2 scalar reference agreement before any posterior "
                    "agreement interpretation"
                ),
            },
            "decision_table": {
                "decision": "Phase 1 finite/acceptance short-chain screen",
                "primary_criterion_status": "passed" if passed else "failed",
                "veto_diagnostic_status": "no vetoes" if passed else f"vetoes: {phase1_vetoes}",
                "main_uncertainty": (
                    "three short CPU-hidden fixed-kernel chains do not establish convergence, "
                    "posterior correctness, tuning quality, zero divergences when native "
                    "divergence is unavailable, or default readiness"
                ),
                "next_justified_action": (
                    "refresh and review Phase 2 reference agreement"
                    if passed
                    else "repair Phase 1 failure before Phase 2"
                ),
                "what_is_not_being_concluded": (
                    "No HMC readiness, convergence, posterior correctness, zero-divergence "
                    "claim when native divergence is unavailable, sampler superiority, "
                    "default readiness, GPU/XLA readiness, or Zhao-Cui source-faithfulness."
                ),
            },
            "run_manifest": {
                "command": (
                    "CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python "
                    "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py "
                    "--json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.json "
                    "--markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.md"
                ),
                "git": base_payload.get("git", {}),
                "environment": base_payload.get("environment", {}),
                "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
                "cpu_gpu_status": "CPU-hidden debug/reference exception",
                "jit_compile": False,
                "tf32_mode": "disabled_by_cpu_hidden_debug_contract",
                "native_divergence_telemetry_status": gate["native_divergence_statuses"],
                "native_divergence_interpretation": gate["native_divergence_interpretation"],
                "zero_divergence_claim_made": False,
                "data_version": "stateless_simulated_scalar_ssl_lstm_filtering_path_v1",
                "random_seeds": settings.seeds,
                "wall_time_seconds": float(time.perf_counter() - start),
                "output_artifacts": (
                    str(DEFAULT_JSON_PATH.relative_to(ROOT)),
                    str(DEFAULT_MARKDOWN_PATH.relative_to(ROOT)),
                ),
                "plan_file": PLAN_PATH,
                "subplan_file": SUBPLAN_PATH,
                "result_file": RESULT_PATH,
            },
            "nonclaims": NONCLAIMS,
        }
    )
    return base.json_ready(base_payload)


def phase1_gate_policy() -> Mapping[str, Any]:
    return {
        "seed_count": 3,
        "acceptance_lower_exclusive": 0.05,
        "acceptance_upper_exclusive": 0.99,
        "native_divergence_policy": (
            "positive native divergence is a hard veto when available; unavailable native "
            "divergence is recorded as unavailable and is not zero-divergence evidence"
        ),
        "log_accept_policy": "finite/nonfinite only; threshold tails are not native divergence telemetry",
    }


def evaluate_phase1_gate(
    seed_rows: Sequence[Mapping[str, Any]],
    *,
    expected_seed_count: int,
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    acceptance_rates: list[float] = []
    native_statuses: list[str] = []
    native_positive_count = 0
    native_available_count = 0
    native_unavailable_count = 0

    if len(seed_rows) != int(expected_seed_count):
        vetoes.append("seed_row_count_mismatch")

    for seed_index, row in enumerate(seed_rows):
        prefix = f"seed_{seed_index}"
        if row.get("status") != "passed_short_smoke":
            vetoes.append(f"{prefix}_short_smoke_failed")
        if row.get("hmc_error") is not None:
            vetoes.append(f"{prefix}_hmc_runtime_error")
        for hard_veto in row.get("hard_vetoes", ()):
            vetoes.append(f"{prefix}_{hard_veto}")

        samples = row.get("samples_summary", {})
        finite_sample_count = int(samples.get("finite_sample_count", 0))
        nonfinite_sample_count = int(samples.get("nonfinite_sample_count", 1))
        if finite_sample_count <= 0 or nonfinite_sample_count != 0:
            vetoes.append(f"{prefix}_retained_samples_nonfinite_or_missing")

        trace = row.get("trace_summary", {})
        target = trace.get("target_log_prob", {})
        if not isinstance(target, Mapping) or target.get("finite") is not True:
            vetoes.append(f"{prefix}_target_log_prob_nonfinite_or_missing")

        log_accept = trace.get("log_accept_ratio", {})
        if (
            not isinstance(log_accept, Mapping)
            or int(log_accept.get("finite_count", 0)) <= 0
            or int(log_accept.get("nonfinite_count", 1)) != 0
        ):
            vetoes.append(f"{prefix}_log_accept_ratio_nonfinite_or_missing")

        acceptance = trace.get("acceptance_rate")
        if acceptance is None or not np.isfinite(float(acceptance)):
            vetoes.append(f"{prefix}_acceptance_missing_or_nonfinite")
        else:
            acceptance_value = float(acceptance)
            acceptance_rates.append(acceptance_value)
            if not (0.05 < acceptance_value < 0.99):
                vetoes.append(f"{prefix}_acceptance_outside_phase1_screen")

        native = trace.get("native_divergence", {})
        if isinstance(native, Mapping) and native.get("available") is True:
            native_available_count += 1
            count = int(native.get("count", 0))
            native_positive_count += count
            native_statuses.append("available")
            if count > 0:
                vetoes.append(f"{prefix}_native_divergence_detected")
        else:
            native_unavailable_count += 1
            if isinstance(native, Mapping):
                native_statuses.append(str(native.get("status", "unavailable")))
            else:
                native_statuses.append("unavailable")

    if native_positive_count > 0:
        native_interpretation = "positive native divergence detected"
    elif native_available_count == len(seed_rows) and len(seed_rows) == int(expected_seed_count):
        native_interpretation = "native divergence available with zero positive indicators"
    else:
        native_interpretation = (
            "native divergence unavailable for at least one seed; unavailable is not zero divergences"
        )

    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "seed_count": len(seed_rows),
        "expected_seed_count": int(expected_seed_count),
        "passed_seed_count": sum(1 for row in seed_rows if row.get("status") == "passed_short_smoke"),
        "acceptance_rates": acceptance_rates,
        "acceptance_min": min(acceptance_rates) if acceptance_rates else None,
        "acceptance_max": max(acceptance_rates) if acceptance_rates else None,
        "native_divergence_statuses": native_statuses,
        "native_divergence_available_count": native_available_count,
        "native_divergence_unavailable_count": native_unavailable_count,
        "native_divergence_positive_count": native_positive_count,
        "native_divergence_interpretation": native_interpretation,
        "zero_divergence_claim_made": False,
        "log_accept_threshold_used_as_native_divergence": False,
    }


def telemetry_policy_payload(gate: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "native_divergence_statuses": gate.get("native_divergence_statuses", ()),
        "native_divergence_interpretation": gate.get("native_divergence_interpretation"),
        "zero_divergence_claim_made": False,
        "unavailable_native_divergence_is_zero_divergence": False,
        "log_accept_threshold_used_as_native_divergence": False,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    decision = payload["decision"]
    gate = payload.get("phase1_gate", {})
    aggregate = payload.get("aggregate_summary", {})
    lines = [
        "# Scalar SSL-LSTM Filtering HMC Validation Phase 1 - 2026-07-09",
        "",
        "## Decision",
        "",
        f"- phase1_short_chain_screen_passed: `{decision['phase1_short_chain_screen_passed']}`",
        f"- vetoes: `{decision['vetoes']}`",
        f"- passed_seed_count: `{decision['passed_seed_count']}` / `{decision['seed_count']}`",
        f"- zero_divergence_claim_made: `{decision['zero_divergence_claim_made']}`",
        f"- next_justified_action: {decision['next_justified_action']}",
        "",
        "## Phase 1 Gate",
        "",
        f"- acceptance rates: `{gate.get('acceptance_rates')}`",
        f"- acceptance range: `{gate.get('acceptance_min')}` to `{gate.get('acceptance_max')}`",
        f"- native divergence statuses: `{gate.get('native_divergence_statuses')}`",
        f"- native divergence interpretation: {gate.get('native_divergence_interpretation')}",
        f"- log-accept threshold used as native divergence: `{gate.get('log_accept_threshold_used_as_native_divergence')}`",
        "",
        "## Aggregate Summary",
        "",
        f"- max abs u by seed: `{aggregate.get('max_abs_u_by_seed')}`",
        f"- target log-prob overall range: `{aggregate.get('target_log_prob_min_overall')}` to `{aggregate.get('target_log_prob_max_overall')}`",
        f"- log-accept max abs by seed: `{aggregate.get('log_accept_max_abs_by_seed')}`",
        f"- interpretation: {aggregate.get('statistical_interpretation')}",
        "",
        "## Seed Rows",
        "",
        "| seed index | seed | status | vetoes | acceptance | finite samples | native divergence |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload.get("seed_rows", ()):
        trace = row.get("trace_summary", {})
        samples = row.get("samples_summary", {})
        lines.append(
            f"| {row.get('seed_index')} | {row.get('seed')} | {row.get('status')} | "
            f"{', '.join(row.get('hard_vetoes', ())) or 'none'} | "
            f"{trace.get('acceptance_rate')} | {samples.get('finite_sample_count')} | "
            f"{trace.get('native_divergence')} |"
        )
    lines.extend(
        [
            "",
            "## Inference Status",
            "",
            "| field | value |",
            "| --- | --- |",
        ]
    )
    for key, value in payload["inference_status"].items():
        lines.append(f"| {key} | {value} |")
    manifest = payload.get("run_manifest", {})
    lines.extend(
        [
            "",
            "## Run Manifest",
            "",
            "| field | value |",
            "| --- | --- |",
            f"| command | `{manifest.get('command')}` |",
            f"| git | `{manifest.get('git')}` |",
            f"| environment | `{manifest.get('environment')}` |",
            f"| conda_env | `{manifest.get('conda_env')}` |",
            f"| cpu_gpu_status | {manifest.get('cpu_gpu_status')} |",
            f"| jit_compile | `{manifest.get('jit_compile')}` |",
            f"| tf32_mode | {manifest.get('tf32_mode')} |",
            f"| native_divergence_telemetry_status | `{manifest.get('native_divergence_telemetry_status')}` |",
            f"| native_divergence_interpretation | {manifest.get('native_divergence_interpretation')} |",
            f"| random_seeds | `{manifest.get('random_seeds')}` |",
            f"| wall_time_seconds | `{manifest.get('wall_time_seconds')}` |",
            f"| output_artifacts | `{manifest.get('output_artifacts')}` |",
            f"| plan_file | `{manifest.get('plan_file')}` |",
            f"| subplan_file | `{manifest.get('subplan_file')}` |",
            f"| result_file | `{manifest.get('result_file')}` |",
        ]
    )
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {item}" for item in payload["nonclaims"])
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--geometry-json", type=Path, default=DEFAULT_GEOMETRY_PATH)
    parser.add_argument("--mass-json", type=Path, default=DEFAULT_MASS_PATH)
    parser.add_argument("--phase4-json", type=Path, default=DEFAULT_PHASE4_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MARKDOWN_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_phase1_validation(
        load_json(args.geometry_json),
        load_json(args.mass_json),
        load_json(args.phase4_json),
    )
    payload["source_artifacts"] = {
        "geometry_json": str(args.geometry_json),
        "mass_json": str(args.mass_json),
        "phase4_json": str(args.phase4_json),
    }
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
