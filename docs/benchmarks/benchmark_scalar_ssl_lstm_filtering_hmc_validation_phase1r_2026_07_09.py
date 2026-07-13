"""Phase 1R scalar SSL-LSTM filtering HMC acceptance-envelope repair.

This harness repeats the Phase 1 finite/acceptance screen with the same fixed
kernel, trajectory length, and seeds, but more retained draws.  It tests whether
the Phase 1 all-accepted 16-draw seed is persistent without changing thresholds,
tuning the kernel, or making posterior/readiness claims.
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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCRIPT_NAME = "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py"
SCHEMA_VERSION = "scalar_ssl_lstm.filtering_hmc_validation_phase1r.v1"
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md"
)
SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase1r-acceptance-envelope-repair-subplan-2026-07-09.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase1r-acceptance-envelope-repair-result-2026-07-09.md"
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
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json"
)
DEFAULT_MARKDOWN_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.md"
)
BASE_REPLICATED_MODULE_PATH = (
    ROOT / "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_replicated_diagnostic_2026_07_08.py"
)
PHASE1_MODULE_PATH = (
    ROOT / "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py"
)
NONCLAIMS = (
    "Phase 1R finite/acceptance repair screen only",
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


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_base_module() -> Any:
    return load_module(
        BASE_REPLICATED_MODULE_PATH,
        "scalar_ssl_lstm_filtering_hmc_replicated_diagnostic_phase1r_reuse",
    )


def load_phase1_module() -> Any:
    return load_module(
        PHASE1_MODULE_PATH,
        "scalar_ssl_lstm_filtering_hmc_validation_phase1_for_phase1r",
    )


def phase1r_settings(base_module: Any) -> Any:
    return base_module.ReplicatedDiagnosticSettings(
        num_leapfrog_steps=4,
        step_size=0.3925,
        num_results=64,
        num_burnin_steps=4,
        seeds=(
            (20260709, 6101),
            (20260709, 6102),
            (20260709, 6103),
        ),
    )


def load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_phase1r_validation(
    geometry_payload: Mapping[str, Any],
    mass_payload: Mapping[str, Any],
    phase4_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    start = time.perf_counter()
    base = load_base_module()
    phase1 = load_phase1_module()
    settings = phase1r_settings(base)
    base_payload = dict(
        base.run_replicated_diagnostic(
            geometry_payload,
            mass_payload,
            phase4_payload,
            settings=settings,
        )
    )
    gate = phase1.evaluate_phase1_gate(
        base_payload.get("seed_rows", ()),
        expected_seed_count=len(settings.seeds),
    )
    base_decision = dict(base_payload.get("decision", {}))
    base_vetoes = tuple(base_decision.get("vetoes", ()))
    phase1r_vetoes = tuple(dict.fromkeys((*base_vetoes, *gate["vetoes"])))
    passed = bool(not phase1r_vetoes and base_decision.get("replicated_diagnostic_passed") is True)

    base_payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_role": "cpu_hidden_scalar_filtering_hmc_validation_phase1r",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "script": f"docs/benchmarks/{SCRIPT_NAME}",
            "plan_path": PLAN_PATH,
            "subplan_path": SUBPLAN_PATH,
            "result_path": RESULT_PATH,
            "settings": {
                **dict(base_payload.get("settings", {})),
                "phase1r_repair_policy": phase1r_repair_policy(),
                "phase1_gate_policy": phase1.phase1_gate_policy(),
            },
            "phase1r_gate": gate,
            "telemetry_policy": phase1.telemetry_policy_payload(gate),
            "decision": {
                "phase1r_acceptance_repair_screen_passed": passed,
                "vetoes": phase1r_vetoes,
                "base_replicated_diagnostic_passed": base_decision.get(
                    "replicated_diagnostic_passed"
                ),
                "base_replicated_vetoes": base_vetoes,
                "passed_seed_count": gate["passed_seed_count"],
                "seed_count": gate["seed_count"],
                "zero_divergence_claim_made": False,
                "viable_for_phase2_reference_agreement_subplan": passed,
                "next_justified_action": (
                    "write Phase 1R result and refresh/review Phase 2 reference-agreement subplan"
                    if passed
                    else "write Phase 1R blocker/next-repair result before Phase 2"
                ),
            },
            "metric_roles": {
                "phase1r_acceptance_repair_screen_passed": "primary_phase1r_pass_fail",
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
                "hmc_readiness": "not assessed; Phase 1R finite/acceptance repair screen only",
                "next_evidence_needed": (
                    "reviewed Phase 2 scalar reference agreement before any posterior "
                    "agreement interpretation"
                    if passed
                    else "reviewed fixed-trajectory integration-envelope ladder or blocker"
                ),
            },
            "decision_table": {
                "decision": "Phase 1R longer same-kernel finite/acceptance repair screen",
                "primary_criterion_status": "passed" if passed else "failed",
                "veto_diagnostic_status": (
                    "no vetoes" if passed else f"vetoes: {phase1r_vetoes}"
                ),
                "main_uncertainty": (
                    "three short CPU-hidden fixed-kernel chains do not establish convergence, "
                    "posterior correctness, tuning quality, zero divergences when native "
                    "divergence is unavailable, or default readiness"
                ),
                "next_justified_action": (
                    "refresh and review Phase 2 reference agreement"
                    if passed
                    else "write a blocker/next-repair result before Phase 2"
                ),
                "what_is_not_being_concluded": (
                    "No HMC readiness, convergence, posterior correctness, zero-divergence "
                    "claim when native divergence is unavailable, sampler superiority, "
                    "default readiness, GPU/XLA readiness, or Zhao-Cui source-faithfulness."
                ),
            },
            "run_manifest": {
                "command": (
                    "CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 720 python "
                    "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py "
                    "--json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json "
                    "--markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.md"
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


def phase1r_repair_policy() -> Mapping[str, Any]:
    return {
        "repair_trigger": "Phase 1 seed (20260709, 6103) acceptance was 1.0 with 16 retained draws",
        "settings_changed": ("num_results",),
        "settings_held_fixed": (
            "num_leapfrog_steps",
            "step_size",
            "trajectory_length_L_times_epsilon",
            "seeds",
            "num_burnin_steps",
            "mass_matrix",
            "target",
            "acceptance_thresholds",
        ),
        "num_results": 64,
        "num_burnin_steps": 4,
        "no_tuning": True,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    decision = payload["decision"]
    gate = payload.get("phase1r_gate", {})
    aggregate = payload.get("aggregate_summary", {})
    lines = [
        "# Scalar SSL-LSTM Filtering HMC Validation Phase 1R - 2026-07-09",
        "",
        "## Decision",
        "",
        f"- phase1r_acceptance_repair_screen_passed: `{decision['phase1r_acceptance_repair_screen_passed']}`",
        f"- vetoes: `{decision['vetoes']}`",
        f"- passed_seed_count: `{decision['passed_seed_count']}` / `{decision['seed_count']}`",
        f"- zero_divergence_claim_made: `{decision['zero_divergence_claim_made']}`",
        f"- next_justified_action: {decision['next_justified_action']}",
        "",
        "## Phase 1R Gate",
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
    payload = run_phase1r_validation(
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
