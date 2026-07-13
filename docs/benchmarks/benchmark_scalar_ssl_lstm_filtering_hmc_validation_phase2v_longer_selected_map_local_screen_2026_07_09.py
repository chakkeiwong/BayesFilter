"""Phase 2V longer selected MAP-local scalar SSL-LSTM HMC screen.

This diagnostic reruns only the Phase 2U selected MAP-local fixed kernel with
more retained draws.  It is a CPU-hidden finite/acceptance screen.  It does not
claim posterior correctness, HMC readiness, convergence, zero divergences,
GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCRIPT_NAME = (
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_2026_07_09.py"
)
SCHEMA_VERSION = "scalar_ssl_lstm.filtering_hmc_validation_phase2v_longer_selected_map_local_screen.v1"
PLAN_PATH = "docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md"
SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2v-longer-selected-map-local-screen-subplan-2026-07-09.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2v-longer-selected-map-local-screen-result-2026-07-09.md"
)
DEFAULT_PHASE2S_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2T_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2U_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.json"
)
DEFAULT_JSON_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json"
)
DEFAULT_MARKDOWN_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.md"
)
PHASE2U_MODULE_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py"
)
NONCLAIMS = (
    "Phase 2V longer selected MAP-local finite/acceptance screen only",
    "not HMC readiness evidence",
    "not HMC convergence evidence",
    "not posterior correctness evidence",
    "not a zero-divergence claim when native divergence is unavailable",
    "not sampler superiority evidence",
    "not statistically supported ranking evidence",
    "not GPU/XLA production-readiness evidence",
    "not default-readiness evidence",
    "not Zhao-Cui source-faithfulness evidence",
)


@dataclass(frozen=True)
class Phase2VScreenSettings:
    """Fixed Phase 2V selected-kernel screen settings."""

    num_leapfrog_steps: int = 2
    step_size: float = 0.785
    num_results: int = 128
    num_burnin_steps: int = 8
    seed: tuple[int, int] = (20260709, 6401)
    acceptance_lower_exclusive: float = 0.05
    acceptance_upper_exclusive: float = 0.99

    def __post_init__(self) -> None:
        for name in ("num_leapfrog_steps", "num_results", "num_burnin_steps"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        step = float(self.step_size)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be positive finite")
        object.__setattr__(self, "step_size", step)
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain exactly two integers")
        object.__setattr__(self, "seed", seed)
        lower = float(self.acceptance_lower_exclusive)
        upper = float(self.acceptance_upper_exclusive)
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError("acceptance envelope must satisfy 0 <= lower < upper <= 1")
        object.__setattr__(self, "acceptance_lower_exclusive", lower)
        object.__setattr__(self, "acceptance_upper_exclusive", upper)

    @property
    def trajectory_length(self) -> float:
        return float(self.num_leapfrog_steps * self.step_size)

    def payload(self) -> Mapping[str, Any]:
        return {
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "step_size": self.step_size,
            "trajectory_length_L_times_epsilon": self.trajectory_length,
            "num_results": self.num_results,
            "num_burnin_steps": self.num_burnin_steps,
            "seed": self.seed,
            "initial_state_u_new": (0.0, 0.0, 0.0, 0.0),
            "chain_execution_mode": "eager",
            "use_xla": False,
            "adaptation_policy": "fixed_kernel_no_adaptation",
            "acceptance_envelope": {
                "lower_exclusive": self.acceptance_lower_exclusive,
                "upper_exclusive": self.acceptance_upper_exclusive,
            },
        }


def load_phase2u_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase2u_for_phase2v",
        PHASE2U_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Phase 2U harness module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_phase2v_longer_selected_map_local_screen(
    phase2s_payload: Mapping[str, Any],
    phase2t_payload: Mapping[str, Any],
    phase2u_payload: Mapping[str, Any],
    settings: Phase2VScreenSettings | None = None,
) -> Mapping[str, Any]:
    cfg = Phase2VScreenSettings() if settings is None else settings
    start = time.perf_counter()
    phase2u = load_phase2u_module()
    precondition = validate_phase2u_handoff(phase2u_payload, cfg)
    vetoes = list(precondition.get("vetoes", ()))
    adapter = None
    adapter_audit: Mapping[str, Any] = {"built": False, "vetoes": ()}
    row: Mapping[str, Any] | None = None
    if not vetoes:
        adapter, adapter_audit = phase2u.build_phase2u_adapter(phase2s_payload)
        vetoes.extend(adapter_audit.get("vetoes", ()))
        if adapter is None:
            vetoes.append("phase2v_adapter_not_built")

    if adapter is not None and not adapter_audit.get("vetoes"):
        initial_state_check = validate_initial_state_zero(adapter)
        vetoes.extend(initial_state_check.get("vetoes", ()))
        row = phase2u.run_candidate(
            adapter,
            settings=_phase2u_compatible_settings(cfg),
            candidate_index=0,
            num_leapfrog_steps=cfg.num_leapfrog_steps,
            step_size=cfg.step_size,
        )
    else:
        initial_state_check = {"passed": False, "vetoes": ("phase2v_adapter_missing",)}
        vetoes.append("longer_selected_screen_not_run")

    gate = evaluate_phase2v_gate(row, cfg)
    vetoes.extend(gate.get("vetoes", ()))
    unique_vetoes = tuple(dict.fromkeys(vetoes))
    passed = bool(not unique_vetoes and gate.get("passed") is True)
    telemetry = phase2u.telemetry_policy_payload(
        [] if row is None else [row],
        phase2s_payload,
        phase2t_payload,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "cpu_hidden_scalar_filtering_hmc_phase2v_longer_selected_map_local_screen",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "script": f"docs/benchmarks/{SCRIPT_NAME}",
        "plan_path": PLAN_PATH,
        "subplan_path": SUBPLAN_PATH,
        "result_path": RESULT_PATH,
        "classification": "extension_or_invention",
        "target_scope": None if adapter is None else adapter.target_scope,
        "settings": cfg.payload(),
        "source_artifacts": {
            "phase2s_json": str(DEFAULT_PHASE2S_PATH.relative_to(ROOT)),
            "phase2t_json": str(DEFAULT_PHASE2T_PATH.relative_to(ROOT)),
            "phase2u_json": str(DEFAULT_PHASE2U_PATH.relative_to(ROOT)),
        },
        "precondition": precondition,
        "adapter_audit": adapter_audit,
        "initial_state_check": initial_state_check,
        "selected_kernel_row": row,
        "phase2v_gate": gate,
        "telemetry_policy": telemetry,
        "environment": environment_payload(),
        "git": git_payload(),
        "decision": {
            "phase2v_longer_selected_map_local_screen_passed": passed,
            "vetoes": unique_vetoes,
            "selected_kernel": {
                "num_leapfrog_steps": cfg.num_leapfrog_steps,
                "step_size": cfg.step_size,
                "trajectory_length_L_times_epsilon": cfg.trajectory_length,
                "phase2u_selected_candidate_index": 0,
            },
            "acceptance_rate": None if row is None else row.get("acceptance_rate"),
            "zero_divergence_claim_made": False,
            "viable_for_scalar_reference_posterior_agreement_subplan": passed,
            "viable_for_phase3_gpu_xla_subplan": False,
            "next_justified_action": (
                "write Phase 2V result and draft/review scalar reference/posterior-agreement subplan"
                if passed
                else "write Phase 2V blocker or narrower tuning/localization repair result"
            ),
        },
        "metric_roles": {
            "phase2v_longer_selected_map_local_screen_passed": "primary_phase2v_pass_fail",
            "initial_state_u_new_zero": "hard_veto_evidence",
            "retained_sample_finiteness": "hard_veto_evidence",
            "target_log_prob_finiteness": "hard_veto_evidence",
            "log_accept_ratio_finiteness": "hard_veto_evidence",
            "acceptance_envelope": "phase2v_pass_fail_gate",
            "native_divergence": "hard_veto_if_available_positive; unavailable is not zero divergences",
            "acceptance_value": "descriptive_after_screen",
            "runtime": "explanatory_only",
        },
        "inference_status": {
            "hard_veto_screen": "passed" if passed else "failed",
            "native_divergence": telemetry["native_divergence_interpretation"],
            "zero_divergence_claim": "not made",
            "statistically_supported_ranking": "none; single selected-kernel screen",
            "descriptive_only_differences": (
                "acceptance, target-log-prob range, log-accept range, sample range, and runtime"
            ),
            "posterior_correctness": "not assessed",
            "hmc_readiness": "not assessed; Phase 2V finite/acceptance screen only",
            "gpu_xla_readiness": "blocked",
            "default_readiness": "not assessed",
            "next_evidence_needed": (
                "reviewed scalar reference/posterior-agreement subplan"
                if passed
                else "reviewed narrower tuning/localization repair"
            ),
        },
        "decision_table": {
            "decision": "Phase 2V longer selected MAP-local fixed-kernel HMC screen",
            "primary_criterion_status": "passed" if passed else "failed",
            "veto_diagnostic_status": "no vetoes" if passed else f"vetoes: {unique_vetoes}",
            "main_uncertainty": (
                "A 128-draw CPU-hidden finite/acceptance screen cannot establish "
                "posterior correctness, convergence, zero divergences, or GPU/default readiness."
            ),
            "next_justified_action": (
                "draft/review scalar reference/posterior-agreement subplan"
                if passed
                else "write blocker or narrower repair"
            ),
            "what_is_not_being_concluded": (
                "No posterior correctness, HMC readiness, convergence, zero-divergence "
                "claim when native divergence is unavailable, sampler superiority, "
                "statistical ranking, GPU/XLA readiness, default readiness, or "
                "Zhao-Cui source faithfulness."
            ),
        },
        "run_manifest": {
            "command": (
                "CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python "
                "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_2026_07_09.py "
                "--json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json "
                "--markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.md"
            ),
            "git": git_payload(),
            "environment": environment_payload(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "cpu_gpu_status": "CPU-hidden debug/reference exception",
            "jit_compile": False,
            "tf32_mode": "disabled_by_cpu_hidden_debug_contract",
            "data_version": "stateless_simulated_scalar_ssl_lstm_filtering_path_v1",
            "random_seeds": (cfg.seed,),
            "wall_time_seconds": float(time.perf_counter() - start),
            "output_artifacts": (
                str(DEFAULT_JSON_PATH.relative_to(ROOT)),
                str(DEFAULT_MARKDOWN_PATH.relative_to(ROOT)),
            ),
            "plan_file": PLAN_PATH,
            "subplan_file": SUBPLAN_PATH,
            "result_file": RESULT_PATH,
        },
        "post_run_red_team": {
            "strongest_alternative_explanation": (
                "A passing longer screen may still only reflect local finite/acceptance "
                "behavior near the MAP-local center, not posterior validity."
            ),
            "what_would_overturn": (
                "Reference disagreement, nonfinite telemetry, positive native divergence "
                "when available, or GPU/XLA mismatch under a reviewed later phase."
            ),
            "weakest_evidence": (
                "Single CPU-hidden selected-kernel screen with unavailable native "
                "divergence telemetry and no posterior reference comparison."
            ),
        },
        "nonclaims": NONCLAIMS,
    }
    return phase2u.json_ready(payload)


def _phase2u_compatible_settings(cfg: Phase2VScreenSettings) -> Any:
    phase2u = load_phase2u_module()
    return phase2u.Phase2UScreenSettings(
        candidate_grid=((cfg.num_leapfrog_steps, cfg.step_size),),
        num_results=cfg.num_results,
        num_burnin_steps=cfg.num_burnin_steps,
        seed_base=cfg.seed,
        acceptance_lower_exclusive=cfg.acceptance_lower_exclusive,
        acceptance_upper_exclusive=cfg.acceptance_upper_exclusive,
    )


def validate_phase2u_handoff(
    phase2u_payload: Mapping[str, Any],
    settings: Phase2VScreenSettings | None = None,
) -> Mapping[str, Any]:
    cfg = Phase2VScreenSettings() if settings is None else settings
    vetoes: list[str] = []
    if (
        phase2u_payload.get("schema_version")
        != "scalar_ssl_lstm.filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen.v1"
    ):
        vetoes.append("phase2u_schema_mismatch")
    decision = phase2u_payload.get("decision", {})
    if decision.get("phase2u_retuned_map_local_hmc_screen_passed") is not True:
        vetoes.append("phase2u_decision_not_passed")
    if decision.get("vetoes"):
        vetoes.append("phase2u_vetoes_present")
    selected = decision.get("selected_candidate") or {}
    if int(selected.get("candidate_index", -1)) != 0:
        vetoes.append("phase2u_selected_candidate_index_mismatch")
    if int(selected.get("num_leapfrog_steps", -1)) != cfg.num_leapfrog_steps:
        vetoes.append("phase2u_selected_leapfrog_mismatch")
    if abs(float(selected.get("step_size", np.nan)) - cfg.step_size) > 1.0e-12:
        vetoes.append("phase2u_selected_step_size_mismatch")
    if (
        abs(
            float(selected.get("trajectory_length_L_times_epsilon", np.nan))
            - cfg.trajectory_length
        )
        > 1.0e-12
    ):
        vetoes.append("phase2u_selected_trajectory_length_mismatch")
    rows = phase2u_payload.get("candidate_rows", ())
    selected_rows = [
        row for row in rows
        if int(row.get("candidate_index", -1)) == 0
    ]
    if len(selected_rows) != 1:
        vetoes.append("phase2u_selected_candidate_row_missing")
        initial = None
    else:
        initial = selected_rows[0].get("initial", {}).get("u_new")
        if not _is_zero_initial_state(initial):
            vetoes.append("phase2u_selected_initial_state_not_zero")
    if decision.get("viable_for_phase3_gpu_xla_subplan") is True:
        vetoes.append("phase2u_gpu_xla_viability_unexpected")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "phase2u_decision": decision,
        "selected_candidate": selected,
        "selected_candidate_initial_u_new": initial,
    }


def validate_initial_state_zero(adapter: Any) -> Mapping[str, Any]:
    vetoes: list[str] = []
    initial = tf.zeros((adapter.parameter_dim,), dtype=tf.float64)
    initial_np = np.asarray(initial.numpy(), dtype=float)
    if initial_np.shape != (4,) or not np.all(initial_np == 0.0):
        vetoes.append("initial_state_u_new_not_zero")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "initial_state_u_new": initial_np,
    }


def evaluate_phase2v_gate(
    row: Mapping[str, Any] | None,
    settings: Phase2VScreenSettings,
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    if row is None:
        return {
            "passed": False,
            "vetoes": ("selected_kernel_row_missing",),
            "acceptance_rate": None,
        }
    for hard_veto in row.get("hard_vetoes", ()):
        vetoes.append(f"selected_kernel_{hard_veto}")
    if row.get("status") != "passed_hard_vetoes":
        vetoes.append("selected_kernel_hard_veto_screen_failed")
    initial = row.get("initial", {}).get("u_new")
    if not _is_zero_initial_state(initial):
        vetoes.append("selected_kernel_initial_state_not_zero")
    acceptance = row.get("acceptance_rate")
    if acceptance is None or not np.isfinite(float(acceptance)):
        vetoes.append("selected_kernel_acceptance_missing_or_nonfinite")
        in_envelope = False
    else:
        in_envelope = bool(
            settings.acceptance_lower_exclusive
            < float(acceptance)
            < settings.acceptance_upper_exclusive
        )
        if not in_envelope:
            vetoes.append("selected_kernel_acceptance_outside_envelope")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "acceptance_rate": acceptance,
        "acceptance_in_envelope": in_envelope,
        "acceptance_envelope": {
            "lower_exclusive": settings.acceptance_lower_exclusive,
            "upper_exclusive": settings.acceptance_upper_exclusive,
        },
    }


def _is_zero_initial_state(value: Any) -> bool:
    array = np.asarray(value, dtype=float)
    return bool(array.shape == (4,) and np.all(np.isfinite(array)) and np.all(array == 0.0))


def environment_payload() -> Mapping[str, Any]:
    return {
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
        "tf_physical_devices": [
            {"name": device.name, "device_type": device.device_type}
            for device in tf.config.list_physical_devices()
        ],
        "tf_logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
    }


def git_payload() -> Mapping[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    try:
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    except Exception:  # noqa: BLE001
        status = ""
    lines = [line for line in status.splitlines() if line.strip()]
    return {
        "commit": commit,
        "dirty": bool(lines),
        "dirty_line_count": len(lines),
        "dirty_preview": lines[:20],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    decision = payload["decision"]
    row = payload.get("selected_kernel_row") or {}
    trace = row.get("trace_summary", {})
    samples = row.get("samples_summary", {})
    lines = [
        "# Scalar SSL-LSTM Filtering HMC Validation Phase 2V - Longer Selected MAP-Local Screen",
        "",
        "## Decision",
        "",
        f"- phase2v_longer_selected_map_local_screen_passed: `{decision['phase2v_longer_selected_map_local_screen_passed']}`",
        f"- vetoes: `{decision['vetoes']}`",
        f"- selected_kernel: `{decision['selected_kernel']}`",
        f"- acceptance_rate: `{decision['acceptance_rate']}`",
        f"- zero_divergence_claim_made: `{decision['zero_divergence_claim_made']}`",
        f"- next_justified_action: {decision['next_justified_action']}",
        "",
        "## Selected Kernel Row",
        "",
        f"- status: `{row.get('status')}`",
        f"- hard vetoes: `{row.get('hard_vetoes')}`",
        f"- initial: `{row.get('initial')}`",
        f"- acceptance: `{row.get('acceptance_rate')}`",
        f"- samples summary: `{samples}`",
        f"- log accept: `{trace.get('log_accept_ratio')}`",
        f"- target log prob: `{trace.get('target_log_prob')}`",
        f"- native divergence: `{trace.get('native_divergence')}`",
        "",
        "## Inference Status",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
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
    parser.add_argument("--phase2s-json", type=Path, default=DEFAULT_PHASE2S_PATH)
    parser.add_argument("--phase2t-json", type=Path, default=DEFAULT_PHASE2T_PATH)
    parser.add_argument("--phase2u-json", type=Path, default=DEFAULT_PHASE2U_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MARKDOWN_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    phase2u = load_phase2u_module()
    payload = run_phase2v_longer_selected_map_local_screen(
        load_json(args.phase2s_json),
        load_json(args.phase2t_json),
        load_json(args.phase2u_json),
    )
    payload["source_artifacts"] = {
        "phase2s_json": str(args.phase2s_json),
        "phase2t_json": str(args.phase2t_json),
        "phase2u_json": str(args.phase2u_json),
    }
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(phase2u.json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
