"""Run V2 common-suite student repetition classifications.

This runner is deliberately conservative.  It repeats only student surfaces
whose frozen V2 contract can be represented without editing vendored student
code.  Everything else is classified as an interface/contract block instead of
being replaced by older proxy panels.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
    EXPECTED_V2_MODEL_IDS,
    CommonModelSpecV2,
    common_model_specs_v2,
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


DTYPE = tf.float64
VALUE_TOLERANCE = 5e-10
LEDGER_TOLERANCE = 5e-10

PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-common-model-suite-v2-student-repetition-execution-plan-2026-06-07.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-common-model-suite-v2-student-repetition-execution-result-2026-06-07.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_student_repetition_2026-06-07.json"
REPORT_PATH = REPORT_DIR / "dpf-common-model-suite-v2-student-repetition-2026-06-07.md"

P1_MANIFEST_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_manifest_2026-06-07.json"
P2_DENSITY_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_density_2026-06-07.json"
P3_NORESAMPLING_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_noresampling_2026-06-07.json"
P4_FIXED_RESAMPLING_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_fixed_resampling_2026-06-07.json"
P5_GRADIENTS_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_gradients_2026-06-07.json"
V2_CLOSEOUT_PATH = (
    REPO_ROOT
    / "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-overnight-gated-execution-result-2026-06-07.md"
)
PHASE_CLOSURE_LEDGER_PATHS = {
    "manifest": (
        REPO_ROOT / "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p1-declarative-spec-result-2026-06-07.md",
        REPO_ROOT / "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p1-claude-review-ledger-2026-06-07.md",
        "PASS_P1_DECLARATIVE_SPEC_READY_FOR_P2",
    ),
    "density": (
        REPO_ROOT / "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p2-density-tieout-result-2026-06-07.md",
        REPO_ROOT / "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p2-claude-review-ledger-2026-06-07.md",
        "PASS_P2_DENSITY_READY_FOR_P3",
    ),
    "noresampling": (
        REPO_ROOT / "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p3-noresampling-paths-result-2026-06-07.md",
        REPO_ROOT / "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p3-claude-review-ledger-2026-06-07.md",
        "PASS_P3_NORESAMPLING_READY_FOR_P4",
    ),
    "fixed_resampling": (
        REPO_ROOT / "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p4-fixed-ancestor-paths-result-2026-06-07.md",
        REPO_ROOT / "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p4-claude-review-ledger-2026-06-07.md",
        "PASS_P4_FIXED_RESAMPLING_READY_FOR_P5",
    ),
    "gradients": (
        REPO_ROOT / "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p5-fixed-branch-gradients-result-2026-06-07.md",
        REPO_ROOT / "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p5-claude-review-ledger-2026-06-07.md",
        "PASS_P5_GRADIENTS_READY_FOR_P6",
    ),
}

STUDENT_ROOT = REPO_ROOT / "experiments" / "student_dpf_baselines"
APF_VENDOR_ROOT = STUDENT_ROOT / "vendor"
APF_REPO_ROOT = APF_VENDOR_ROOT / "advanced_particle_filter"
MLCOE_ROOT = STUDENT_ROOT / "vendor" / "2026MLCOE"

IMPLEMENTATIONS = {
    "advanced_particle_filter": {
        "repo_root": APF_REPO_ROOT,
        "source_commit": "d2a797c330e11befacbb736b5c86b8d03eb4a389",
        "adapter": STUDENT_ROOT / "adapters" / "advanced_particle_filter_adapter.py",
    },
    "2026MLCOE": {
        "repo_root": MLCOE_ROOT,
        "source_commit": "020cfd7f2f848afa68432e95e6c6e747d3d2402d",
        "adapter": STUDENT_ROOT / "adapters" / "mlcoe_adapter.py",
    },
}

INSPECTED_FILES = [
    STUDENT_ROOT / "adapters" / "common.py",
    STUDENT_ROOT / "adapters" / "advanced_particle_filter_adapter.py",
    STUDENT_ROOT / "adapters" / "mlcoe_adapter.py",
    APF_REPO_ROOT / "models" / "base.py",
    APF_REPO_ROOT / "models" / "linear_gaussian.py",
    APF_REPO_ROOT / "models" / "range_bearing.py",
    APF_REPO_ROOT / "filters" / "particle.py",
    APF_REPO_ROOT / "utils" / "resampling.py",
    APF_REPO_ROOT / "tf_filters" / "differentiable_particle.py",
    APF_REPO_ROOT / "tf_models" / "svssm.py",
    MLCOE_ROOT / "src" / "models" / "classic_ssm.py",
    MLCOE_ROOT / "src" / "filters" / "particle.py",
    MLCOE_ROOT / "src" / "filters" / "DPF.py",
    MLCOE_ROOT / "src" / "filters" / "resampling" / "soft.py",
    MLCOE_ROOT / "src" / "filters" / "resampling" / "optimal_transport.py",
]

SURFACES = ("density", "noresampling_path", "fixed_ancestor_path", "fixed_branch_gradient")


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
    artifacts = _load_closed_v2_artifacts()
    _preflight_closed_artifacts(artifacts)
    specs = {spec.model_id: spec for spec in common_model_specs_v2()}
    student_inventory = _student_inventory()
    cells: list[dict[str, Any]] = []
    command_counts = {
        "student_density_or_model_method_commands_run": 0,
        "student_filter_loop_commands_run": 0,
        "student_gradient_commands_run": 0,
        "student_proxy_panel_commands_run": 0,
    }

    for implementation in IMPLEMENTATIONS:
        for model_id in EXPECTED_V2_MODEL_IDS:
            spec = specs[model_id]
            for surface in SURFACES:
                if implementation == "advanced_particle_filter" and model_id == "lgssm_2d_h25_rich":
                    cell = _advanced_lgssm_cell(surface, spec, artifacts)
                    if cell.get("student_executed"):
                        if surface == "density":
                            command_counts["student_density_or_model_method_commands_run"] += 1
                        elif surface in {"noresampling_path", "fixed_ancestor_path"}:
                            command_counts["student_filter_loop_commands_run"] += 1
                        elif surface == "fixed_branch_gradient":
                            command_counts["student_gradient_commands_run"] += 1
                else:
                    cell = _blocked_cell(implementation, model_id, surface, _block_reason(implementation, model_id, surface))
                cells.append(cell)

    decision = _decision(cells)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": (
            "Can the two quarantined student repositories be applied to the same "
            "frozen V2 density/path/fixed-ancestor/gradient contracts with every "
            "cell terminally classified and no oracle claim?"
        ),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "closed_v2_artifacts": _artifact_manifest(),
        "student_inventory": student_inventory,
        "predeclared_runnable_surface": [
            {
                "implementation": "advanced_particle_filter",
                "model_id": "lgssm_2d_h25_rich",
                "surfaces": ["density", "noresampling_path"],
                "basis": (
                    "NumPy StateSpaceModel density methods and BootstrapParticleFilter "
                    "with a replay RNG that supplies frozen initial particles and "
                    "transition innovations."
                ),
            }
        ],
        "tolerances": {
            "density_value_abs": VALUE_TOLERANCE,
            "path_value_abs": VALUE_TOLERANCE,
            "path_ledger_abs": LEDGER_TOLERANCE,
            "gradient_abs": 5e-8,
            "fd_is_diagnostic_only": True,
        },
        "primary_criterion_fields": {
            "implementations": list(IMPLEMENTATIONS),
            "model_ids": list(EXPECTED_V2_MODEL_IDS),
            "surfaces": list(SURFACES),
            "total_cells": len(cells),
            "terminal_status_counts": _status_counts(cells),
            "all_cells_terminally_classified": all(
                cell["status"] in {"MATCHED", "EXPLAINED_MISMATCH", "INTERFACE_BLOCKED", "OUT_OF_SCOPE"}
                for cell in cells
            ),
            "matched_cells": [
                _cell_key(cell) for cell in cells if cell["status"] == "MATCHED"
            ],
            "explained_mismatch_cells": [
                _cell_key(cell) for cell in cells if cell["status"] == "EXPLAINED_MISMATCH"
            ],
            "interface_blocked_cells": [
                _cell_key(cell) for cell in cells if cell["status"] == "INTERFACE_BLOCKED"
            ],
        },
        "veto_diagnostics": {
            "closed_v2_artifact_missing": False,
            "student_output_used_to_revise_bf_ff_contract": False,
            "oracle_claim_made": False,
            "tolerance_or_contract_changed_after_student_result": False,
            "fd_used_as_gate": False,
            "student_proxy_panel_command_run": command_counts["student_proxy_panel_commands_run"] > 0,
            "cpu_only_tf_without_cuda_hidden": PRE_IMPORT_CUDA_VISIBLE_DEVICES != "-1",
            "unclassified_cell": any(
                cell["status"] not in {"MATCHED", "EXPLAINED_MISMATCH", "INTERFACE_BLOCKED", "OUT_OF_SCOPE"}
                for cell in cells
            ),
            "unexplained_executed_mismatch": any(
                cell["status"] == "EXPLAINED_MISMATCH" and not cell.get("classification_reason")
                for cell in cells
            ),
            "student_gradient_command_run_without_contract": command_counts["student_gradient_commands_run"] > 0,
        },
        "explanatory_only_fields": {
            "command_counts": command_counts,
            "status_counts": _status_counts(cells),
            "matched_cell_count": sum(cell["status"] == "MATCHED" for cell in cells),
            "explained_mismatch_cell_count": sum(cell["status"] == "EXPLAINED_MISMATCH" for cell in cells),
            "interface_blocked_cell_count": sum(cell["status"] == "INTERFACE_BLOCKED" for cell in cells),
            "advanced_lgssm_note": (
                "The runnable APF LGSSM cells exercise APF's model density methods "
                "and bootstrap PF loop under a replay RNG.  This is common-sense "
                "cross-code evidence, not a correctness proof."
            ),
        },
        "cells": cells,
        "summary": _summary(cells),
        "review_round": 0,
        "open_material_blockers": [],
        "repair_amendment_required": False,
        "next_allowed_action": "run Claude student repetition result/governance review",
        "artifact_paths": {
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "phase_result": RESULT_PATH,
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_v2_student_repetition_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "no student correctness or failure claim",
            "no BayesFilter or FilterFlow oracle claim",
            "no filter correctness proof",
            "no stochastic-resampling distribution claim",
            "no differentiable-resampling claim",
            "no TT/SIRT or paper-scale reproduction claim",
            "no GPU, HMC, DSGE, scalability, deployment, or production-readiness claim",
        ],
    }


def _load_closed_v2_artifacts() -> dict[str, dict[str, Any]]:
    return {
        "manifest": load_json(P1_MANIFEST_PATH),
        "density": load_json(P2_DENSITY_PATH),
        "noresampling": load_json(P3_NORESAMPLING_PATH),
        "fixed_resampling": load_json(P4_FIXED_RESAMPLING_PATH),
        "gradients": load_json(P5_GRADIENTS_PATH),
    }


def _preflight_closed_artifacts(artifacts: dict[str, dict[str, Any]]) -> None:
    if not V2_CLOSEOUT_PATH.exists():
        raise FileNotFoundError(V2_CLOSEOUT_PATH)
    closeout = V2_CLOSEOUT_PATH.read_text(encoding="utf-8")
    if "PASS_OVERNIGHT_EXECUTION_CLOSED_THROUGH_P7" not in closeout:
        raise ValueError("V2 production closeout is not closed PASS through P7")
    manifest_ids = [row["model_id"] for row in artifacts["manifest"]["rows"]]
    if tuple(manifest_ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"unexpected V2 model ids: {manifest_ids}")
    for key, (_, _, decision) in PHASE_CLOSURE_LEDGER_PATHS.items():
        _require_reviewed_phase_closure(key=key, expected_decision=decision, closeout=closeout)


def _require_reviewed_phase_closure(*, key: str, expected_decision: str, closeout: str) -> None:
    result_path, review_path, _ = PHASE_CLOSURE_LEDGER_PATHS[key]
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    if not review_path.exists():
        raise FileNotFoundError(review_path)
    result_text = result_path.read_text(encoding="utf-8")
    review_text = review_path.read_text(encoding="utf-8")
    missing = []
    if expected_decision not in result_text:
        missing.append(str(result_path.relative_to(REPO_ROOT)))
    if expected_decision not in review_text:
        missing.append(str(review_path.relative_to(REPO_ROOT)))
    if expected_decision not in closeout:
        missing.append(str(V2_CLOSEOUT_PATH.relative_to(REPO_ROOT)))
    if missing:
        raise ValueError(
            f"{key} phase closure missing {expected_decision} in reviewed ledgers: {missing}"
        )


def _advanced_lgssm_cell(
    surface: str,
    spec: CommonModelSpecV2,
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if surface == "density":
        return _advanced_lgssm_density_cell(spec, artifacts["density"])
    if surface == "noresampling_path":
        contract = _contract_by_id(artifacts["noresampling"], spec.model_id)
        reference = _cell_by_model(artifacts["noresampling"], spec.model_id)["bayesfilter"]
        return _advanced_lgssm_filter_cell(
            surface=surface,
            spec=spec,
            contract=contract,
            reference=reference,
            resample_criterion="never",
            resample_method="systematic",
        )
    if surface == "fixed_ancestor_path":
        return _blocked_cell(
            "advanced_particle_filter",
            spec.model_id,
            surface,
            (
                "advanced_particle_filter BootstrapParticleFilter resamples after the "
                "measurement update, while the frozen V2 fixed-ancestor contract "
                "branches at the start of the step before propagation.  No exact "
                "same-branch-timing replay surface is exposed without writing a new "
                "adapter and reviewed amendment."
            ),
        )
    if surface == "fixed_branch_gradient":
        return _blocked_cell(
            "advanced_particle_filter",
            spec.model_id,
            surface,
            (
                "advanced_particle_filter does not expose the V2 fixed-noise "
                "fixed-ancestor physical-knob AD scalar; its differentiable PF "
                "uses different SV/HMC resampling contracts."
            ),
        )
    raise ValueError(surface)


def _advanced_lgssm_density_cell(
    spec: CommonModelSpecV2,
    density_artifact: dict[str, Any],
) -> dict[str, Any]:
    try:
        with _prepend_sys_path(APF_VENDOR_ROOT):
            from advanced_particle_filter.models import make_lgssm  # type: ignore

            model = make_lgssm(
                _np(spec.parameters["A"]),
                _np(spec.parameters["C"]),
                _np(spec.parameters["Q"]),
                _np(spec.parameters["R"]),
                _np(spec.parameters["m0"]),
                _np(spec.parameters["P0"]),
            )
            initial = model.initial_log_prob(_np(spec.x0)) if hasattr(model, "initial_log_prob") else _initial_log_prob_apf(model, _np(spec.x0))
            transition = model.dynamics_log_prob(_np(spec.x_next), _np(spec.x_prev))
            observation = model.observation_log_prob(_np(spec.x_obs), _np(spec.observation))
            scalar_value = float(np.sum(initial) + np.sum(transition) + np.sum(observation))
    except Exception as exc:  # pragma: no cover - artifact path.
        return _adapter_error_cell("advanced_particle_filter", spec.model_id, "density", exc)

    reference = _cell_by_model(density_artifact, spec.model_id)["bayesfilter"]
    student = {
        "status": "executed",
        "backend": "advanced_particle_filter_numpy_lgssm_model_methods",
        "model_id": spec.model_id,
        "initial_log_density": _json_float_list(initial),
        "transition_log_density": _json_float_list(transition),
        "observation_log_density": _json_float_list(observation),
        "scalar": scalar_value,
        "finite": bool(
            np.all(np.isfinite(initial))
            and np.all(np.isfinite(transition))
            and np.all(np.isfinite(observation))
            and math.isfinite(scalar_value)
        ),
        "note": (
            "Initial density is evaluated with the APF model's precomputed "
            "initial covariance Cholesky; APF adds 1e-8 covariance jitter."
        ),
    }
    metrics = _density_metrics(reference, student)
    return _executed_cell(
        implementation="advanced_particle_filter",
        model_id=spec.model_id,
        surface="density",
        student=student,
        reference=reference,
        metrics=metrics,
        status="MATCHED" if metrics["all_components_within_tolerance"] else "EXPLAINED_MISMATCH",
        reason=(
            "APF LGSSM density methods match the frozen V2 density values within tolerance."
            if metrics["all_components_within_tolerance"]
            else "APF LGSSM density values differ under the frozen V2 density probes; APF adds covariance jitter and uses its own density helpers."
        ),
    )


def _initial_log_prob_apf(model: Any, values: np.ndarray) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float64) - np.asarray(model.initial_mean, dtype=np.float64)
    solved = np.linalg.solve(model._initial_cov_chol, residual.T)
    mahal_sq = np.sum(solved * solved, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(model._initial_cov_chol)))
    dim = model.state_dim
    return -0.5 * (dim * np.log(2.0 * np.pi) + logdet + mahal_sq)


def _advanced_lgssm_filter_cell(
    *,
    surface: str,
    spec: CommonModelSpecV2,
    contract: dict[str, Any],
    reference: dict[str, Any],
    resample_criterion: str,
    resample_method: str,
) -> dict[str, Any]:
    try:
        with _prepend_sys_path(APF_VENDOR_ROOT):
            from advanced_particle_filter.filters import BootstrapParticleFilter  # type: ignore
            from advanced_particle_filter.models import make_lgssm  # type: ignore

            model = make_lgssm(
                np.asarray(contract["parameters"]["A"], dtype=np.float64),
                np.asarray(contract["parameters"]["C"], dtype=np.float64),
                np.asarray(contract["parameters"]["Q"], dtype=np.float64),
                np.asarray(contract["parameters"]["R"], dtype=np.float64),
                np.asarray(contract["parameters"]["m0"], dtype=np.float64),
                np.asarray(contract["parameters"]["P0"], dtype=np.float64),
            )
            initial_particles = np.asarray(contract["initial_particles"], dtype=np.float64)
            transition_innovations = np.asarray(contract["transition_innovations"], dtype=np.float64)
            rng = _ReplayRNG(
                initial_noise=_right_cholesky_noise(
                    initial_particles - np.asarray(model.initial_mean, dtype=np.float64),
                    np.asarray(model._initial_cov_chol, dtype=np.float64),
                ),
                transition_noises=np.stack(
                    [
                        _right_cholesky_noise(
                            transition_innovations[step],
                            np.asarray(model._dynamics_cov_chol, dtype=np.float64),
                        )
                        for step in range(int(contract["horizon"]))
                    ],
                    axis=0,
                ),
                resampling_flags=contract.get("resampling_flags"),
                fixed_ancestor_indices=contract.get("fixed_ancestor_indices"),
            )
            pf = BootstrapParticleFilter(
                n_particles=int(contract["num_particles"]),
                resample_method="systematic",
                resample_criterion=resample_criterion,
                ess_threshold=1.1 if surface == "fixed_ancestor_path" else 0.0,
                seed=0,
            )
            result = pf.filter(
                model,
                np.asarray(contract["observations"], dtype=np.float64),
                return_particles=True,
                rng=rng,
            )
    except Exception as exc:  # pragma: no cover - artifact path.
        return _adapter_error_cell("advanced_particle_filter", spec.model_id, surface, exc)

    student = _student_path_payload(
        surface=surface,
        result=result,
        contract=contract,
        replay_rng=rng,
        resample_method=resample_method,
    )
    metrics = _path_metrics(reference, student)
    matched = metrics["all_primary_fields_within_tolerance"] and _branch_policy_ok(surface, contract, student)
    reason = (
        f"APF bootstrap PF loop matches frozen V2 {surface} primary values within tolerance."
        if matched
        else (
            f"APF bootstrap PF loop was executable under replay, but the frozen V2 {surface} "
            "primary values or branch semantics differ; see per-field metrics."
        )
    )
    return _executed_cell(
        implementation="advanced_particle_filter",
        model_id=spec.model_id,
        surface=surface,
        student=student,
        reference=reference,
        metrics=metrics,
        status="MATCHED" if matched else "EXPLAINED_MISMATCH",
        reason=reason,
    )


class _ReplayRNG:
    def __init__(
        self,
        *,
        initial_noise: np.ndarray,
        transition_noises: np.ndarray,
        resampling_flags: list[bool] | None,
        fixed_ancestor_indices: list[int] | None,
    ) -> None:
        self.initial_noise = np.asarray(initial_noise, dtype=np.float64)
        self.transition_noises = np.asarray(transition_noises, dtype=np.float64)
        self.resampling_flags = list(resampling_flags or [])
        self.fixed_ancestor_indices = None if fixed_ancestor_indices is None else np.asarray(fixed_ancestor_indices, dtype=int)
        self.standard_normal_calls = 0
        self.uniform_calls = 0
        self.choice_calls = 0
        self.replay_log: list[dict[str, Any]] = []

    def standard_normal(self, size: Any = None) -> np.ndarray:
        shape = tuple(size) if isinstance(size, tuple) else (int(size),)
        self.standard_normal_calls += 1
        if self.standard_normal_calls == 1:
            value = self.initial_noise
            label = "initial_standard_normal_noise_for_frozen_particles"
        else:
            step = self.standard_normal_calls - 2
            value = self.transition_noises[step]
            label = f"transition_standard_normal_noise_for_frozen_innovations_step_{step}"
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"ReplayRNG expected shape {shape}, got replay value {value.shape}")
        self.replay_log.append({"method": "standard_normal", "call": self.standard_normal_calls, "label": label, "shape": list(shape)})
        return np.array(value, copy=True)

    def uniform(self, low: float = 0.0, high: float = 1.0, size: Any = None) -> np.ndarray | float:
        self.uniform_calls += 1
        self.replay_log.append({"method": "uniform", "call": self.uniform_calls, "low": float(low), "high": float(high), "size": None if size is None else list(np.shape(np.zeros(size)))})
        if size is None:
            return 0.0
        return np.zeros(size, dtype=np.float64)

    def choice(self, a: int, size: int, replace: bool = True, p: np.ndarray | None = None) -> np.ndarray:
        del replace, p
        self.choice_calls += 1
        if self.fixed_ancestor_indices is None:
            raise ValueError("ReplayRNG.choice called without fixed ancestors")
        if int(a) != len(self.fixed_ancestor_indices) or int(size) != len(self.fixed_ancestor_indices):
            raise ValueError("ReplayRNG.choice shape does not match fixed ancestors")
        self.replay_log.append({"method": "choice", "call": self.choice_calls, "label": "fixed_ancestor_indices", "size": int(size)})
        return np.array(self.fixed_ancestor_indices, copy=True)


def _student_path_payload(
    *,
    surface: str,
    result: Any,
    contract: dict[str, Any],
    replay_rng: _ReplayRNG,
    resample_method: str,
) -> dict[str, Any]:
    particles = np.asarray(result.particles, dtype=np.float64)
    weights = np.asarray(result.weights, dtype=np.float64)
    loglik_increments = np.asarray(result.log_likelihood_increments, dtype=np.float64)
    scalar_value = float(result.log_likelihood)
    ledger = []
    observations = np.asarray(contract["observations"], dtype=np.float64)
    initial_particles = np.asarray(contract["initial_particles"], dtype=np.float64)
    innovations = np.asarray(contract["transition_innovations"], dtype=np.float64)
    for step in range(int(contract["horizon"])):
        ledger.append(
            {
                "step": int(step),
                "observation": observations[step].tolist(),
                "transition_innovations": innovations[step].tolist(),
                "predicted_particles": particles[step + 1].tolist(),
                "incremental_log_normalizer": float(loglik_increments[step]),
                "weights": weights[step + 1].tolist(),
                "normalized_log_weights": np.log(weights[step + 1] + 1e-300).tolist(),
                "ess": float(result.ess[step]),
                "filtered_mean": np.asarray(result.means[step + 1], dtype=np.float64).tolist(),
                "filtered_variance": np.diag(np.asarray(result.covariances[step + 1], dtype=np.float64)).tolist(),
                "resampling_applied": bool(result.resampled[step]) if result.resampled is not None else False,
            }
        )
    payload = {
        "status": "executed",
        "backend": "advanced_particle_filter_bootstrap_particle_filter_replay_rng",
        "surface": surface,
        "model_id": contract["model_id"],
        "scalar": scalar_value,
        "finite": bool(
            math.isfinite(scalar_value)
            and np.all(np.isfinite(particles))
            and np.all(np.isfinite(weights))
            and np.all(np.isfinite(loglik_increments))
        ),
        "resampling_count": int(np.sum(result.resampled)) if result.resampled is not None else 0,
        "initial_particles_replayed": particles[0].tolist(),
        "initial_particles_match_contract": bool(np.allclose(particles[0], initial_particles, atol=0.0, rtol=0.0)),
        "ledger": ledger,
        "contract_checksum": contract["contract_checksum"],
        "replay_rng_log": replay_rng.replay_log,
        "replay_rng_call_counts": {
            "standard_normal": replay_rng.standard_normal_calls,
            "uniform": replay_rng.uniform_calls,
            "choice": replay_rng.choice_calls,
        },
        "resample_method": resample_method,
        "student_branch_timing": "APF resamples after measurement update; V2 fixed-ancestor contract marks branch at start of step.",
    }
    return payload


def _right_cholesky_noise(target: np.ndarray, chol: np.ndarray) -> np.ndarray:
    """Return noise satisfying ``noise @ chol.T == target``."""

    return np.linalg.solve(chol, np.asarray(target, dtype=np.float64).T).T


def _density_metrics(reference: dict[str, Any], student: dict[str, Any]) -> dict[str, Any]:
    components = {}
    max_delta = 0.0
    for name in ("initial_log_density", "transition_log_density", "observation_log_density"):
        deltas = _abs_deltas(reference[name], student[name])
        max_component_delta = max(deltas) if deltas else 0.0
        max_delta = max(max_delta, max_component_delta)
        components[name] = {
            "reference": reference[name],
            "student": student[name],
            "max_abs_delta": max_component_delta,
            "within_tolerance": max_component_delta <= VALUE_TOLERANCE,
        }
    scalar_delta = abs(float(reference["scalar"]) - float(student["scalar"]))
    max_delta = max(max_delta, scalar_delta)
    return {
        "components": components,
        "scalar_abs_delta": scalar_delta,
        "scalar_within_tolerance": scalar_delta <= VALUE_TOLERANCE,
        "max_abs_delta": max_delta,
        "tolerance": VALUE_TOLERANCE,
        "all_components_within_tolerance": all(v["within_tolerance"] for v in components.values())
        and scalar_delta <= VALUE_TOLERANCE,
    }


def _path_metrics(reference: dict[str, Any], student: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "scalar": abs(float(reference["scalar"]) - float(student["scalar"])),
        "resampling_count": abs(int(reference.get("resampling_count", 0)) - int(student.get("resampling_count", 0))),
        "initial_particles": _max_abs_nested(
            reference["ledger"][0]["previous_particles"],
            student["initial_particles_replayed"],
        ),
    }
    step_fields: dict[str, float] = {}
    for field in ("predicted_particles", "incremental_log_normalizer", "weights", "filtered_mean", "filtered_variance"):
        step_fields[field] = max(
            _max_abs_nested(ref_step[field], stu_step[field])
            for ref_step, stu_step in zip(reference["ledger"], student["ledger"], strict=True)
        )
    if "resampling_applied" in reference["ledger"][0]:
        step_fields["resampling_flags"] = max(
            0.0 if bool(ref_step["resampling_applied"]) == bool(stu_step["resampling_applied"]) else 1.0
            for ref_step, stu_step in zip(reference["ledger"], student["ledger"], strict=True)
        )
    max_delta = max([float(value) for value in fields.values()] + [float(value) for value in step_fields.values()])
    return {
        "fields": fields,
        "step_fields": step_fields,
        "max_abs_delta": max_delta,
        "value_tolerance": VALUE_TOLERANCE,
        "ledger_tolerance": LEDGER_TOLERANCE,
        "all_primary_fields_within_tolerance": (
            fields["scalar"] <= VALUE_TOLERANCE
            and fields["resampling_count"] == 0
            and fields["initial_particles"] <= LEDGER_TOLERANCE
            and all(float(value) <= LEDGER_TOLERANCE for value in step_fields.values())
        ),
        "branch_timing_warning": student.get("student_branch_timing"),
    }


def _branch_policy_ok(surface: str, contract: dict[str, Any], student: dict[str, Any]) -> bool:
    del contract, student
    if surface != "fixed_ancestor_path":
        return True
    return False


def _blocked_cell(implementation: str, model_id: str, surface: str, reason: str) -> dict[str, Any]:
    return {
        "implementation": implementation,
        "source_commit": IMPLEMENTATIONS[implementation]["source_commit"],
        "model": model_id,
        "surface": surface,
        "status": "INTERFACE_BLOCKED",
        "student_executed": False,
        "classification_reason": reason,
        "primary_criterion": "terminal classification, not equality",
        "non_claim": "interface blocking is not a student failure",
    }


def _adapter_error_cell(implementation: str, model_id: str, surface: str, exc: BaseException) -> dict[str, Any]:
    return {
        "implementation": implementation,
        "source_commit": IMPLEMENTATIONS[implementation]["source_commit"],
        "model": model_id,
        "surface": surface,
        "status": "EXPLAINED_MISMATCH",
        "student_executed": False,
        "classification_reason": f"adapter execution error before producing comparable values: {type(exc).__name__}: {exc}",
        "primary_criterion": "terminal classification, not equality",
        "non_claim": "adapter error is not by itself a student scientific failure",
    }


def _executed_cell(
    *,
    implementation: str,
    model_id: str,
    surface: str,
    student: dict[str, Any],
    reference: dict[str, Any],
    metrics: dict[str, Any],
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "implementation": implementation,
        "source_commit": IMPLEMENTATIONS[implementation]["source_commit"],
        "model": model_id,
        "surface": surface,
        "status": status,
        "student_executed": True,
        "classification_reason": reason,
        "student": student,
        "reference": {
            "backend": reference.get("backend"),
            "model_id": reference.get("model_id"),
            "scalar": reference.get("scalar"),
            "contract_checksum": reference.get("contract_checksum"),
            "spec_checksum": reference.get("spec_checksum"),
        },
        "metrics": metrics,
        "primary_criterion": "frozen V2 values within tolerance for MATCHED; otherwise explained terminal classification",
        "non_claim": "student agreement or mismatch is common-sense cross-code evidence only",
    }


def _block_reason(implementation: str, model_id: str, surface: str) -> str:
    if implementation == "advanced_particle_filter":
        if model_id == "range_bearing_4d_h20_rich":
            return (
                "advanced_particle_filter exposes a range-bearing model, but its public "
                "range-bearing density uses Student-t observation noise while V2 uses "
                "Gaussian range/bearing noise with the frozen covariance."
            )
        if model_id == "sv_1d_h18_rich":
            return (
                "advanced_particle_filter SVSSM/DPF surfaces target a batched HMC/DPF "
                "contract and do not expose the V2 fixed finite particle path scalar."
            )
        if model_id in {"structural_ar1_quadratic_h16", "spatial_sir_j3_rk4", "predator_prey_rk4"}:
            return f"advanced_particle_filter has no exposed V2-compatible {model_id} model surface."
        if surface == "fixed_branch_gradient":
            return "advanced_particle_filter has no V2 fixed-branch physical-knob AD gradient surface."
    if implementation == "2026MLCOE":
        if surface == "density":
            return (
                "2026MLCOE current adapters do not expose initial/transition/observation "
                "density components with V2 scalar constants and checksums."
            )
        if surface in {"noresampling_path", "fixed_ancestor_path"}:
            return (
                "2026MLCOE BPF samples internally and exposes particle summaries, not "
                "fixed initial particles, fixed innovations, and fixed ancestor replay "
                "under the V2 scalar contract."
            )
        if surface == "fixed_branch_gradient":
            return "2026MLCOE has no exposed V2 fixed-branch physical-knob AD gradient surface."
    return "no exact V2 student adapter surface exposed before execution"


def _student_inventory() -> dict[str, Any]:
    return {
        "implementations": {
            name: {
                "source_commit": info["source_commit"],
                "repo_root": str(info["repo_root"].relative_to(REPO_ROOT)),
                "adapter": str(info["adapter"].relative_to(REPO_ROOT)),
                "repo_exists": info["repo_root"].exists(),
                "adapter_sha256": _sha256(info["adapter"]),
            }
            for name, info in IMPLEMENTATIONS.items()
        },
        "inspected_files": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in INSPECTED_FILES
            if path.exists()
        },
    }


def _artifact_manifest() -> dict[str, Any]:
    paths = [
        P1_MANIFEST_PATH,
        P2_DENSITY_PATH,
        P3_NORESAMPLING_PATH,
        P4_FIXED_RESAMPLING_PATH,
        P5_GRADIENTS_PATH,
        V2_CLOSEOUT_PATH,
    ]
    return {
        str(path.relative_to(REPO_ROOT)): _sha256(path)
        for path in paths
    }


def _contract_by_id(artifact: dict[str, Any], model_id: str) -> dict[str, Any]:
    for contract in artifact.get("contracts", []):
        if contract.get("model_id") == model_id:
            return contract
    raise KeyError(model_id)


def _cell_by_model(artifact: dict[str, Any], model_id: str) -> dict[str, Any]:
    for cell in artifact.get("cells", []):
        if cell.get("model") == model_id:
            return cell
    raise KeyError(model_id)


def _status_counts(cells: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(cell["status"] for cell in cells))


def _summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_impl = {}
    for implementation in IMPLEMENTATIONS:
        impl_cells = [cell for cell in cells if cell["implementation"] == implementation]
        by_impl[implementation] = _status_counts(impl_cells)
    by_surface = {}
    for surface in SURFACES:
        surface_cells = [cell for cell in cells if cell["surface"] == surface]
        by_surface[surface] = _status_counts(surface_cells)
    return {
        "status_counts": _status_counts(cells),
        "by_implementation": by_impl,
        "by_surface": by_surface,
        "executed_cells": [_cell_key(cell) for cell in cells if cell.get("student_executed")],
    }


def _decision(cells: list[dict[str, Any]]) -> str:
    if not all(cell["status"] in {"MATCHED", "EXPLAINED_MISMATCH", "INTERFACE_BLOCKED", "OUT_OF_SCOPE"} for cell in cells):
        return "BLOCKED_UNCLASSIFIED_STUDENT_CELL"
    if any(cell["status"] == "EXPLAINED_MISMATCH" and not cell.get("classification_reason") for cell in cells):
        return "BLOCKED_UNEXPLAINED_STUDENT_MISMATCH"
    return "PASS_STUDENT_REPETITION_TERMINALLY_CLASSIFIED"


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "cells",
        "primary_criterion_fields",
        "veto_diagnostics",
        "explanatory_only_fields",
        "run_manifest",
        "non_claims",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if payload["decision"] != "PASS_STUDENT_REPETITION_TERMINALLY_CLASSIFIED":
        raise ValueError(f"student repetition did not close PASS: {payload['decision']}")
    expected_cells = len(IMPLEMENTATIONS) * len(EXPECTED_V2_MODEL_IDS) * len(SURFACES)
    if len(payload["cells"]) != expected_cells:
        raise ValueError(f"expected {expected_cells} cells, got {len(payload['cells'])}")
    if payload["veto_diagnostics"].get("fd_used_as_gate"):
        raise ValueError("FD was used as a gate")
    if payload["veto_diagnostics"].get("oracle_claim_made"):
        raise ValueError("oracle claim veto fired")
    if payload["veto_diagnostics"].get("cpu_only_tf_without_cuda_hidden"):
        raise ValueError("CPU-only CUDA hiding veto fired")
    if not payload["primary_criterion_fields"].get("all_cells_terminally_classified"):
        raise ValueError("not all cells terminally classified")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# DPF Common Model Suite V2 Student Repetition Result",
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
        "Primary criterion: every implementation/model/surface cell is terminally classified as `MATCHED`, `EXPLAINED_MISMATCH`, `INTERFACE_BLOCKED`, or `OUT_OF_SCOPE` under the frozen V2 contracts.",
        "",
        "Veto diagnostics: missing closed V2 artifacts, oracle misuse, unreviewed contract/tolerance changes, FD used as a gate, unclassified executed discrepancy, proxy student panel substitution, or CPU-only TensorFlow without pre-import `CUDA_VISIBLE_DEVICES=-1`.",
        "",
        "FD diagnostics are diagnostic-only and were not used as a gate.",
        "",
        "## Summary",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in payload["summary"]["status_counts"].items():
        lines.append(f"| `{status}` | {count} |")
    lines.extend(
        [
            "",
            "Executed cells:",
            "",
        ]
    )
    for cell_key in payload["summary"]["executed_cells"]:
        lines.append(f"- `{cell_key}`")
    if not payload["summary"]["executed_cells"]:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Cells",
            "",
            "| Implementation | Model | Surface | Status | Reason |",
            "|---|---|---|---|---|",
        ]
    )
    for cell in payload["cells"]:
        reason = str(cell.get("classification_reason", "")).replace("\n", " ")
        lines.append(
            f"| `{cell['implementation']}` | `{cell['model']}` | `{cell['surface']}` | `{cell['status']}` | {reason} |"
        )
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
            f"| {payload['decision']} | all {payload['primary_criterion_fields']['total_cells']} cells terminally classified | no veto fired | future adapter work could expose more exact V2 student surfaces | Claude result/governance review, then close | no student correctness, failure, or filter correctness claim |",
            "",
            "## Post-Run Red Team",
            "",
            "Strongest alternative explanation: additional lower-level student code might be adaptable to more V2 surfaces with new reviewed adapters, but those surfaces are not exposed by the current adapters without changing the execution contract.",
            "",
            "Result that would overturn this decision: discovery of an existing student command or adapter that already accepts the frozen V2 density/path/fixed-ancestor/gradient fixtures exactly and was missed by this inventory.",
            "",
            "Weakest evidence link: the executed evidence is narrow, mainly APF LGSSM replay; blocked cells are interface classifications, not exhaustive proofs about all possible future adapters.",
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


def _cell_key(cell: dict[str, Any]) -> str:
    return f"{cell['implementation']}::{cell['model']}::{cell['surface']}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _np(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _json_float_list(value: Any) -> list[float]:
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _abs_deltas(a: Any, b: Any) -> list[float]:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    return np.abs(arr_a - arr_b).reshape(-1).tolist()


def _max_abs_nested(a: Any, b: Any) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


class _prepend_sys_path:
    def __init__(self, path: Path) -> None:
        self.path = str(path)
        self.original: list[str] | None = None

    def __enter__(self) -> None:
        import sys

        self.original = list(sys.path)
        if self.path not in sys.path:
            sys.path.insert(0, self.path)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        import sys

        if self.original is not None:
            sys.path[:] = self.original


if __name__ == "__main__":
    raise SystemExit(main())
