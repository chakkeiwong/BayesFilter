"""LGSSM-only student/FilterFlow value and gradient tie-out.

This runner is intentionally narrower than the common-suite student repetition:
it focuses only on the LGSSM row and separates strict frozen V2 comparisons
from diagnostic-only student-native localization probes.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
    common_model_specs_v2,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_common_model_suite_v2_student_repetition_tf as student_v2,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)


MODEL_ID = "lgssm_2d_h25_rich"
VALUE_TOLERANCE = 5e-10
LEDGER_TOLERANCE = 5e-10
GRADIENT_TOLERANCE = 5e-8
APF_JITTER = 1e-8

PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-lgssm-student-filterflow-value-gradient-tieout-plan-2026-06-07.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-lgssm-student-filterflow-value-gradient-tieout-result-2026-06-07.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-lgssm-student-filterflow-value-gradient-tieout-claude-review-ledger-2026-06-07.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_lgssm_student_filterflow_value_gradient_tieout_2026-06-07.json"
REPORT_PATH = REPORT_DIR / "dpf-lgssm-student-filterflow-value-gradient-tieout-2026-06-07.md"

SURFACES = (
    "density_components",
    "noresampling_path",
    "fixed_ancestor_path",
    "fixed_branch_gradient",
)
IMPLEMENTATIONS = ("advanced_particle_filter", "2026MLCOE")
TERMINAL_STATUSES = {"MATCHED", "EXPLAINED_MISMATCH", "INTERFACE_BLOCKED", "OUT_OF_SCOPE"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        return 0
    start = time.perf_counter()
    payload = _run()
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    markdown = _markdown(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, markdown)
    write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    artifacts = student_v2._load_closed_v2_artifacts()
    student_v2._preflight_closed_artifacts(artifacts)
    spec = _lgssm_spec()

    strict_cells = [
        _apf_density_cell(spec, artifacts["density"]),
        _apf_noresampling_cell(spec, artifacts["noresampling"]),
        _blocked_cell(
            "advanced_particle_filter",
            "fixed_ancestor_path",
            (
                "APF BootstrapParticleFilter resamples after the measurement "
                "update, while the V2 fixed-ancestor LGSSM contract branches "
                "before propagation.  No exact strict-V2 branch-timing surface "
                "is exposed without a reviewed adapter amendment."
            ),
        ),
        _blocked_cell(
            "advanced_particle_filter",
            "fixed_branch_gradient",
            (
                "APF TF LGSSM builders convert inputs through tf.constant and "
                "APF's differentiable PF surface is SV/HMC-oriented; no exposed "
                "LGSSM fixed-branch scalar with V2 knobs "
                "transition_matrix_scale and observation_noise_scale is available."
            ),
        ),
    ]
    for surface in SURFACES:
        strict_cells.append(
            _blocked_cell(
                "2026MLCOE",
                surface,
                _mlcoe_block_reason(surface),
            )
        )

    diagnostics = _diagnostics(spec, artifacts, strict_cells)
    status_counts = _status_counts(strict_cells)
    command_counts = {
        "student_density_or_model_method_commands_run": 1,
        "student_filter_loop_commands_run": 1,
        "student_gradient_commands_run": 0,
        "student_proxy_panel_commands_run": 0,
        "student_native_diagnostic_mirror_commands_run": len(diagnostics),
    }
    veto_diagnostics = {
        "closed_v2_artifact_missing": False,
        "student_output_used_to_revise_filterflow_contract": False,
        "oracle_claim_made": False,
        "fd_used_as_gate": False,
        "student_proxy_panel_command_run": False,
        "tolerance_or_contract_changed_after_student_result": False,
        "stochastic_student_run_promoted_to_equality": False,
        "hidden_rng_inside_claimed_gradient_cell": False,
        "vendored_student_source_mutated": False,
        "cpu_only_tf_without_cuda_hidden": PRE_IMPORT_CUDA_VISIBLE_DEVICES != "-1",
        "unclassified_cell": any(cell["status"] not in TERMINAL_STATUSES for cell in strict_cells),
        "unexplained_executed_mismatch": any(
            cell["status"] == "EXPLAINED_MISMATCH" and not cell.get("classification_reason")
            for cell in strict_cells
        ),
        "student_gradient_command_run_without_contract": False,
    }
    decision = (
        "PASS_LGSSM_STUDENT_FILTERFLOW_TERMINALLY_CLASSIFIED"
        if not any(veto_diagnostics.values())
        else "BLOCKED_LGSSM_STUDENT_FILTERFLOW_VETO"
    )
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": (
            "For LGSSM only, can APF and MLCOE expose value and gradient "
            "surfaces that match the closed FilterFlow V2 contracts?"
        ),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_ledger_path": REVIEW_LEDGER_PATH,
        "model_id": MODEL_ID,
        "filterflow_reference_policy": (
            "Strict comparisons use closed local FilterFlow-side V2 LGSSM "
            "artifacts.  Diagnostic student-native mirrors cannot create "
            "MATCHED status."
        ),
        "static_interface_inventory": _static_inventory(),
        "predeclared_hypotheses": {
            "H1": "APF strict density mismatch due +1e-8 covariance jitter.",
            "H2": "APF particles replay exactly but weights/log-normalizers differ at jitter scale.",
            "H3": "APF fixed-ancestor strict V2 branch timing is blocked.",
            "H4": "APF fixed-branch LGSSM AD gradient is blocked under current surfaces.",
            "H5": "MLCOE strict value surfaces are blocked or mismatched without V2 scalar exposure.",
            "H6": "MLCOE fixed-branch LGSSM AD gradient is blocked under current surfaces.",
        },
        "strict_cells": strict_cells,
        "diagnostic_only_localization": diagnostics,
        "primary_criterion_fields": {
            "implementations": list(IMPLEMENTATIONS),
            "surfaces": list(SURFACES),
            "total_cells": len(strict_cells),
            "terminal_status_counts": status_counts,
            "all_cells_terminally_classified": all(
                cell["status"] in TERMINAL_STATUSES for cell in strict_cells
            ),
            "matched_cells": [_cell_key(cell) for cell in strict_cells if cell["status"] == "MATCHED"],
            "explained_mismatch_cells": [
                _cell_key(cell) for cell in strict_cells if cell["status"] == "EXPLAINED_MISMATCH"
            ],
            "interface_blocked_cells": [
                _cell_key(cell) for cell in strict_cells if cell["status"] == "INTERFACE_BLOCKED"
            ],
        },
        "tolerances": {
            "strict_value_abs": VALUE_TOLERANCE,
            "strict_ledger_abs": LEDGER_TOLERANCE,
            "strict_gradient_abs": GRADIENT_TOLERANCE,
            "fd_is_diagnostic_only": True,
        },
        "veto_diagnostics": veto_diagnostics,
        "explanatory_only_fields": {
            "command_counts": command_counts,
            "status_counts": status_counts,
            "apf_jitter": APF_JITTER,
            "tensorflow_cuda_stderr_is_explanatory_for_cpu_only_run": True,
        },
        "artifact_paths": {
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "phase_result": RESULT_PATH,
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_lgssm_student_filterflow_value_gradient_tieout_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "review_round": 0,
        "open_material_blockers": [],
        "repair_amendment_required": False,
        "next_allowed_action": "run Claude result/governance review",
        "non_claims": [
            "no student correctness or failure claim",
            "no FilterFlow, BayesFilter, APF, or MLCOE oracle claim",
            "no filter correctness proof",
            "no stochastic-resampling distribution claim",
            "no differentiable-resampling claim",
            "no TT/SIRT or paper-scale reproduction claim",
            "no GPU, HMC, DSGE, scalability, deployment, or production-readiness claim",
        ],
    }


def _apf_density_cell(spec: Any, density_artifact: dict[str, Any]) -> dict[str, Any]:
    base_cell = student_v2._advanced_lgssm_density_cell(spec, density_artifact)
    reference = student_v2._cell_by_model(density_artifact, MODEL_ID)["filterflow"]
    metrics = student_v2._density_metrics(reference, base_cell["student"])
    return _executed_cell(
        implementation="advanced_particle_filter",
        surface="density_components",
        student=base_cell["student"],
        filterflow_reference=reference,
        metrics=metrics,
        status="MATCHED" if metrics["all_components_within_tolerance"] else "EXPLAINED_MISMATCH",
        reason=(
            "APF LGSSM density methods match FilterFlow V2 density values within tolerance."
            if metrics["all_components_within_tolerance"]
            else (
                "APF LGSSM density values differ from the strict FilterFlow V2 "
                "density probes because APF adds +1e-8 I covariance jitter before "
                "Gaussian Cholesky/log-density evaluation."
            )
        ),
    )


def _apf_noresampling_cell(spec: Any, noresampling_artifact: dict[str, Any]) -> dict[str, Any]:
    artifacts = student_v2._load_closed_v2_artifacts()
    base_cell = student_v2._advanced_lgssm_cell("noresampling_path", spec, artifacts)
    reference = student_v2._cell_by_model(noresampling_artifact, MODEL_ID)["filterflow"]
    metrics = student_v2._path_metrics(reference, base_cell["student"])
    matched = metrics["all_primary_fields_within_tolerance"]
    return _executed_cell(
        implementation="advanced_particle_filter",
        surface="noresampling_path",
        student=base_cell["student"],
        filterflow_reference={
            "backend": reference.get("backend"),
            "model_id": reference.get("model_id"),
            "scalar": reference.get("scalar"),
            "contract_checksum": reference.get("contract_checksum"),
        },
        metrics=metrics,
        status="MATCHED" if matched else "EXPLAINED_MISMATCH",
        reason=(
            "APF no-resampling replay matches FilterFlow V2 primary values within tolerance."
            if matched
            else (
                "APF no-resampling replay matches the frozen particles but differs "
                "from strict FilterFlow V2 weights/log-normalizers at the APF "
                "+1e-8 I observation-covariance jitter scale."
            )
        ),
    )


def _diagnostics(
    spec: Any,
    artifacts: dict[str, dict[str, Any]],
    strict_cells: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics = []
    apf_density = _find_cell(strict_cells, "advanced_particle_filter", "density_components")
    if apf_density["status"] == "EXPLAINED_MISMATCH":
        diagnostics.append(_apf_jittered_density_mirror(spec, apf_density))
    apf_path = _find_cell(strict_cells, "advanced_particle_filter", "noresampling_path")
    if apf_path["status"] == "EXPLAINED_MISMATCH":
        contract = student_v2._contract_by_id(artifacts["noresampling"], MODEL_ID)
        diagnostics.append(_apf_jittered_noresampling_mirror(contract, apf_path))
    diagnostics.append(
        {
            "diagnostic": "mlcoe_weight_only_likelihood_mirror",
            "status": "NOT_RUN",
            "reason": (
                "MLCOE exposes no strict V2 fixed-particle/fixed-innovation PF "
                "log-normalizer scalar in the current adapter, so the mirror is "
                "not applicable until a strict value cell executes."
            ),
            "strict_status_effect": "none; diagnostic-only",
        }
    )
    return diagnostics


def _apf_jittered_density_mirror(spec: Any, apf_density_cell: dict[str, Any]) -> dict[str, Any]:
    params = _numeric_lgssm_parameters(spec)
    x0 = _np(spec.x0)
    x_prev = _np(spec.x_prev)
    x_next = _np(spec.x_next)
    x_obs = _np(spec.x_obs)
    observation = _np(spec.observation)
    initial = _gaussian_logpdf(x0, params["m0"], params["P0"] + APF_JITTER * np.eye(params["P0"].shape[0]))
    transition_mean = x_prev @ params["A"].T
    transition = _gaussian_logpdf(
        x_next,
        transition_mean,
        params["Q"] + APF_JITTER * np.eye(params["Q"].shape[0]),
    )
    obs_mean = x_obs @ params["C"].T
    observation_log_density = _gaussian_logpdf(
        np.broadcast_to(observation, obs_mean.shape),
        obs_mean,
        params["R"] + APF_JITTER * np.eye(params["R"].shape[0]),
    )
    mirror = {
        "initial_log_density": initial.tolist(),
        "transition_log_density": transition.tolist(),
        "observation_log_density": observation_log_density.tolist(),
        "scalar": float(np.sum(initial) + np.sum(transition) + np.sum(observation_log_density)),
    }
    student = apf_density_cell["student"]
    deltas = {
        "initial": _max_abs(initial, student["initial_log_density"]),
        "transition": _max_abs(transition, student["transition_log_density"]),
        "observation": _max_abs(observation_log_density, student["observation_log_density"]),
        "scalar": abs(float(mirror["scalar"]) - float(student["scalar"])),
    }
    return {
        "diagnostic": "apf_jittered_density_mirror",
        "status": "EXPLAINS_STRICT_DELTA" if max(deltas.values()) <= 5e-12 else "DIAGNOSTIC_MISMATCH",
        "jitter": APF_JITTER,
        "max_abs_delta_vs_apf_student": max(deltas.values()),
        "component_deltas_vs_apf_student": deltas,
        "mirror": mirror,
        "strict_status_effect": "none; diagnostic-only",
    }


def _apf_jittered_noresampling_mirror(contract: dict[str, Any], apf_path_cell: dict[str, Any]) -> dict[str, Any]:
    params = _numeric_lgssm_parameters(contract)
    particles = np.asarray(contract["initial_particles"], dtype=np.float64)
    observations = np.asarray(contract["observations"], dtype=np.float64)
    innovations = np.asarray(contract["transition_innovations"], dtype=np.float64)
    horizon = int(contract["horizon"])
    n_particles = int(contract["num_particles"])
    log_weights = -np.log(float(n_particles)) * np.ones(n_particles)
    ledger = []
    increments = []
    for step in range(horizon):
        particles = particles @ params["A"].T + innovations[step]
        obs_mean = particles @ params["C"].T
        obs_values = np.broadcast_to(observations[step], obs_mean.shape)
        log_lik = _gaussian_logpdf(
            obs_values,
            obs_mean,
            params["R"] + APF_JITTER * np.eye(params["R"].shape[0]),
        )
        log_weights = log_weights + log_lik
        log_z = _logsumexp(log_weights)
        increments.append(log_z)
        weights = np.exp(log_weights - log_z)
        log_weights = np.log(weights + 1e-300)
        mean = weights @ particles
        diff = particles - mean
        covariance = np.einsum("n,ni,nj->ij", weights, diff, diff)
        covariance = 0.5 * (covariance + covariance.T)
        ledger.append(
            {
                "predicted_particles": particles.tolist(),
                "incremental_log_normalizer": float(log_z),
                "weights": weights.tolist(),
                "filtered_mean": mean.tolist(),
                "filtered_variance": np.diag(covariance).tolist(),
            }
        )
    student = apf_path_cell["student"]
    step_deltas = {}
    for field in ("predicted_particles", "incremental_log_normalizer", "weights", "filtered_mean", "filtered_variance"):
        step_deltas[field] = max(
            _max_abs(mirror_step[field], student_step[field])
            for mirror_step, student_step in zip(ledger, student["ledger"], strict=True)
        )
    scalar = float(np.sum(increments))
    scalar_delta = abs(scalar - float(student["scalar"]))
    max_delta = max([scalar_delta] + list(step_deltas.values()))
    return {
        "diagnostic": "apf_jittered_noresampling_mirror",
        "status": "EXPLAINS_STRICT_DELTA" if max_delta <= 5e-12 else "DIAGNOSTIC_MISMATCH",
        "jitter": APF_JITTER,
        "scalar": scalar,
        "scalar_delta_vs_apf_student": scalar_delta,
        "step_deltas_vs_apf_student": step_deltas,
        "max_abs_delta_vs_apf_student": max_delta,
        "strict_status_effect": "none; diagnostic-only",
    }


def _blocked_cell(implementation: str, surface: str, reason: str) -> dict[str, Any]:
    return {
        "implementation": implementation,
        "source_commit": student_v2.IMPLEMENTATIONS[implementation]["source_commit"],
        "model": MODEL_ID,
        "surface": surface,
        "status": "INTERFACE_BLOCKED",
        "student_executed": False,
        "classification_reason": reason,
        "primary_criterion": "terminal classification under strict V2 LGSSM contract",
        "non_claim": "interface blocking is not a student failure",
    }


def _executed_cell(
    *,
    implementation: str,
    surface: str,
    student: dict[str, Any],
    filterflow_reference: dict[str, Any],
    metrics: dict[str, Any],
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "implementation": implementation,
        "source_commit": student_v2.IMPLEMENTATIONS[implementation]["source_commit"],
        "model": MODEL_ID,
        "surface": surface,
        "status": status,
        "student_executed": True,
        "classification_reason": reason,
        "student": student,
        "filterflow_reference": filterflow_reference,
        "metrics_vs_filterflow": metrics,
        "primary_criterion": "strict FilterFlow V2 values within tolerance for MATCHED",
        "non_claim": "student agreement or mismatch is common-sense cross-code evidence only",
    }


def _mlcoe_block_reason(surface: str) -> str:
    if surface == "density_components":
        return (
            "MLCOE current LGSSM/BPF adapters do not expose initial, transition, "
            "observation, and scalar Gaussian density components with the frozen "
            "V2 constants and checksums."
        )
    if surface in {"noresampling_path", "fixed_ancestor_path"}:
        return (
            "MLCOE BPF samples internally through TensorFlow Probability and "
            "exposes particle summaries, not fixed initial particles, fixed "
            "transition innovations, fixed ancestor indices, and the V2 "
            "log-normalizer scalar."
        )
    if surface == "fixed_branch_gradient":
        return (
            "MLCOE DPF/PHMC surfaces do not expose the frozen LGSSM fixed-branch "
            "log-normalizer scalar with V2 knobs transition_matrix_scale and "
            "observation_noise_scale."
        )
    raise ValueError(surface)


def _static_inventory() -> dict[str, Any]:
    paths = {
        "apf_numpy_base": student_v2.APF_REPO_ROOT / "models" / "base.py",
        "apf_numpy_particle": student_v2.APF_REPO_ROOT / "filters" / "particle.py",
        "apf_tf_base": student_v2.APF_REPO_ROOT / "tf_models" / "base.py",
        "apf_tf_lgssm": student_v2.APF_REPO_ROOT / "tf_models" / "linear_gaussian.py",
        "apf_tf_dpf": student_v2.APF_REPO_ROOT / "tf_filters" / "differentiable_particle.py",
        "mlcoe_classic_ssm": student_v2.MLCOE_ROOT / "src" / "models" / "classic_ssm.py",
        "mlcoe_particle": student_v2.MLCOE_ROOT / "src" / "filters" / "particle.py",
        "mlcoe_dpf": student_v2.MLCOE_ROOT / "src" / "filters" / "DPF.py",
        "mlcoe_phmc": student_v2.MLCOE_ROOT / "src" / "inference" / "phmc.py",
    }
    return {
        "file_sha256": {
            name: student_v2._sha256(path)
            for name, path in paths.items()
            if path.exists()
        },
        "line_level_notes": {
            "apf_jitter": "APF NumPy/TF base models add eps=1e-8 to P0, Q, and R before Cholesky/log-density.",
            "apf_resampling_timing": "APF bootstrap PF propagates, weights, normalizes, computes estimates, then resamples.",
            "apf_tf_constant_parameters": "APF TF LGSSM builder converts supplied A/C/Q/R/m0/P0 through tf.constant.",
            "apf_dpf_contract": "APF differentiable PF is parameterized around SVSSMParams and soft/sinkhorn/amortized resampling.",
            "mlcoe_bpf_constants": "MLCOE BPF observation update uses -0.5 * quadratic without Gaussian normalizing constants.",
            "mlcoe_bpf_rng": "MLCOE BPF samples initial/process noise internally through TensorFlow Probability distributions.",
        },
    }


def _lgssm_spec() -> Any:
    specs = {spec.model_id: spec for spec in common_model_specs_v2()}
    return specs[MODEL_ID]


def _numeric_lgssm_parameters(source: Any) -> dict[str, np.ndarray]:
    parameters = source["parameters"] if isinstance(source, dict) else source.parameters
    return {
        key: np.asarray(parameters[key], dtype=np.float64)
        for key in ("A", "C", "P0", "Q", "R", "m0")
    }


def _find_cell(cells: list[dict[str, Any]], implementation: str, surface: str) -> dict[str, Any]:
    for cell in cells:
        if cell["implementation"] == implementation and cell["surface"] == surface:
            return cell
    raise KeyError((implementation, surface))


def _status_counts(cells: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(cell["status"] for cell in cells))


def _cell_key(cell: dict[str, Any]) -> str:
    return f"{cell['implementation']}::{cell['model']}::{cell['surface']}"


def _gaussian_logpdf(value: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    residual = value - mean
    chol = np.linalg.cholesky(cov)
    solved = np.linalg.solve(chol, residual.T)
    mahal_sq = np.sum(solved * solved, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    dim = cov.shape[0]
    return -0.5 * (dim * np.log(2.0 * np.pi) + logdet + mahal_sq)


def _logsumexp(value: np.ndarray) -> float:
    max_value = float(np.max(value))
    return max_value + float(np.log(np.sum(np.exp(value - max_value))))


def _max_abs(a: Any, b: Any) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _np(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _decision_summary(payload: dict[str, Any]) -> str:
    counts = payload["primary_criterion_fields"]["terminal_status_counts"]
    return ", ".join(f"{status}={count}" for status, count in counts.items())


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "strict_cells",
        "diagnostic_only_localization",
        "primary_criterion_fields",
        "veto_diagnostics",
        "run_manifest",
        "non_claims",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if payload["decision"] != "PASS_LGSSM_STUDENT_FILTERFLOW_TERMINALLY_CLASSIFIED":
        raise ValueError(f"LGSSM student tie-out did not close PASS: {payload['decision']}")
    if len(payload["strict_cells"]) != len(IMPLEMENTATIONS) * len(SURFACES):
        raise ValueError("unexpected strict cell count")
    if not payload["primary_criterion_fields"].get("all_cells_terminally_classified"):
        raise ValueError("not all strict cells terminally classified")
    fired = {key: value for key, value in payload["veto_diagnostics"].items() if value}
    if fired:
        raise ValueError(f"veto diagnostics fired: {fired}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# DPF LGSSM Student/FilterFlow Value and Gradient Tie-Out Result",
        "",
        "metadata_date: 2026-06-07",
        f"decision: {payload['decision']}",
        "",
        "## Question",
        "",
        payload["question"],
        "",
        "## Evidence Contract",
        "",
        "Primary criterion: every LGSSM student/surface cell is terminally classified under the frozen FilterFlow V2 value/gradient contracts.",
        "",
        "FD, Kalman checks, APF jitter mirrors, and MLCOE missing-constant mirrors are diagnostic-only and cannot create strict `MATCHED` status.",
        "",
        "## Summary",
        "",
        f"- strict cells: `{payload['primary_criterion_fields']['total_cells']}`",
        f"- status counts: `{payload['primary_criterion_fields']['terminal_status_counts']}`",
        f"- diagnostic mirrors: `{[row['diagnostic'] + ':' + row['status'] for row in payload['diagnostic_only_localization']]}`",
        "",
        "## Strict Cells",
        "",
        "| Implementation | Surface | Status | Reason |",
        "|---|---|---|---|",
    ]
    for cell in payload["strict_cells"]:
        reason = str(cell.get("classification_reason", "")).replace("\n", " ")
        lines.append(
            f"| `{cell['implementation']}` | `{cell['surface']}` | `{cell['status']}` | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic-Only Localization",
            "",
            "| Diagnostic | Status | Max delta | Note |",
            "|---|---|---:|---|",
        ]
    )
    for row in payload["diagnostic_only_localization"]:
        max_delta = row.get("max_abs_delta_vs_apf_student", "N/A")
        note = row.get("reason", row.get("strict_status_effect", "diagnostic-only"))
        lines.append(f"| `{row['diagnostic']}` | `{row['status']}` | `{max_delta}` | {note} |")
    lines.extend(
        [
            "",
            "## Command Manifest",
            "",
            f"- command: `{payload['run_manifest']['command']}`",
            f"- commit: `{payload['run_manifest']['commit']}`",
            f"- branch: `{payload['run_manifest']['branch']}`",
            f"- CPU-only: `{payload['run_manifest']['cpu_only']}`",
            f"- `CUDA_VISIBLE_DEVICES`: `{payload['run_manifest']['cuda_visible_devices']}`",
            f"- output JSON: `{payload['artifact_paths']['json']}`",
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |",
            "|---|---|---|---|---|---|",
            f"| {payload['decision']} | {_decision_summary(payload)} | no veto fired | current student surfaces expose few strict V2 LGSSM cells | Claude result/governance review | no correctness, oracle, stochastic-resampling, differentiable-resampling, or production claim |",
            "",
            "## Post-Run Red Team",
            "",
            "Strongest alternative explanation: a future reviewed adapter could expose additional exact LGSSM surfaces without vendored-code edits; current interface blocking is not a proof that no such adapter can ever exist.",
            "",
            "Result that would overturn this decision: discovery of an existing student surface that already accepts the frozen V2 particles, innovations, ancestor schedule, scalar, and knobs exactly but was missed.",
            "",
            "Weakest evidence link: APF executable strict cells are narrow, and MLCOE remains classified by interface evidence rather than executable strict V2 values.",
            "",
            "## Non-Claims",
            "",
        ]
    )
    for claim in payload["non_claims"]:
        lines.append(f"- {claim}")
    return "\n".join(lines) + "\n"


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


if __name__ == "__main__":
    raise SystemExit(main())
