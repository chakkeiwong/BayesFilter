#!/usr/bin/env python3
"""Standard-library-only terminal audit for the multi-model NeuTra program."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROGRAM_ID = "multimodel-neutra-filter-posterior-20260715"
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "docs/plans/artifacts" / PROGRAM_ID
P0_REGISTRY = ARTIFACT_ROOT / "phase-p0/attempt-04-20260715T1658/target_registry.json"
PLAN_PATH = REPO_ROOT / (
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p7-synthesis-subplan-2026-07-15.md"
)

EXPECTED_STATES = {
    "SVX-SGQF": "TARGET_BLOCKED_FILTER_ADMISSION",
    "SVX-ZC": "TARGET_BLOCKED_SOURCE_ROUTE_MISMATCH",
    "KSC-UKF": "TARGET_BLOCKED_FILTER_ADMISSION",
    "PP-SGQF": "NEUTRA_CONFIRMED",
    "PP-UKF": "NEUTRA_CONFIRMED",
    "PP-ZC": "TARGET_BLOCKED_SOURCE_ROUTE_MISMATCH",
    "STR-UKF": "COMPARATOR_BLOCKED_GEOMETRY",
    "STR-ZC": "TARGET_BLOCKED_EXTENSION_ROUTE_NOT_DESIGNED",
    "SIR-SGQF": "NEUTRA_CONFIRMED",
    "SIR-UKF": "IMPLEMENTATION_BLOCKED_GPU_SCORE_PARITY",
    "SIR-ZC": "TARGET_BLOCKED_MISSING_OBSERVED_DATA_SCORE_ROUTE",
}

EVIDENCE = {
    "SVX-SGQF": (
        "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
        "phase-p2/SVX-SGQF/target-admission/attempt-04-20260715T103649Z/result.json",
        "344a8f21bfc602f4b88649501003eceb811e02a37607948ceb2813a282513b43",
    ),
    "SVX-ZC": (
        "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
        "phase-p2/SVX-SGQF/target-admission/attempt-04-20260715T103649Z/result.json",
        "344a8f21bfc602f4b88649501003eceb811e02a37607948ceb2813a282513b43",
    ),
    "KSC-UKF": (
        "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
        "phase-p3/KSC-UKF/target-admission/attempt-01-20260715T110415Z/result.json",
        "9398d1536c9629d3dcc6fa98e24ca3d1b214422c59d866279168158eda40a187",
    ),
    "PP-SGQF": (
        "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
        "phase-p4/PP-SGQF/neutra-confirmation/attempt-02/result.json",
        "a77d5edf2b8129d6ff95844e9c5d4bb94b7125c9997777b517f36b830fbda9c4",
    ),
    "PP-UKF": (
        "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
        "phase-p4/PP-UKF/neutra-confirmation/attempt-03/result.json",
        "d9b4f603b28acb06154ab554f41f745c5f544e2516ba4969c6b21d9e5268bacf",
    ),
    "PP-ZC": (
        "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
        "phase-p4/phase-close/attempt-01/cell_ledger.json",
        "7382c76bac8fb877ea971e36ab08ce402b019c488cad9009f5da61dd79dcc87f",
    ),
    "STR-UKF": (
        "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
        "phase-p5/STR-UKF/affine-geometry/attempt-01/result.json",
        "0ee609c414673d3dc3f797aa135ae2de349cec5be39df18241b6de584a5f12d9",
    ),
    "STR-ZC": (
        "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
        "p5-structural-result-2026-07-16.md",
        "b0273e5e0a8218ca472926ca0f9376b3a46a0d627eba5ca2933ad1b99d96be24",
    ),
    "SIR-SGQF": (
        "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
        "phase-p6/SIR-SGQF/neutra-confirmation/attempt-01/result.json",
        "e8b6c159648ade9f2919d97674ffc50a8b55d75d591a256291c3abfdcd4dbcce",
    ),
    "SIR-UKF": (
        "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
        "phase-p6/SIR-common/gpu-canary/attempt-02/gpu_canary.json",
        "51d61ea606521fe553555792ff771c1810424344bdcae2c300e42344731716b9",
    ),
    "SIR-ZC": (
        "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
        "p6-sir-source-support-ledger-2026-07-16.md",
        "fffcd7d92bc21569782fc36ebd9cde05a84eeda416518a9c90b36b2618d58462",
    ),
}

R4_ROOTS = {
    "PP-UKF": ARTIFACT_ROOT / "phase-p4/PP-UKF/neutra-confirmation/attempt-03",
    "PP-SGQF": ARTIFACT_ROOT / "phase-p4/PP-SGQF/neutra-confirmation/attempt-02",
    "SIR-SGQF": ARTIFACT_ROOT / "phase-p6/SIR-SGQF/neutra-confirmation/attempt-01",
}

ACTIVE_STATIC_PATHS = (
    "bayesfilter/inference/neutra_batching.py",
    "bayesfilter/inference/neutra_training.py",
    "bayesfilter/inference/neutra_hmc.py",
    "bayesfilter/inference/neutra_campaign.py",
    "docs/benchmarks/run_multimodel_neutra_p4_predator_prey_training.py",
    "docs/benchmarks/run_multimodel_neutra_p4_predator_prey_neutra_confirmation.py",
    "docs/benchmarks/run_multimodel_neutra_p6_sir_sgqf_training.py",
    "docs/benchmarks/run_multimodel_neutra_p6_sir_sgqf_neutra_confirmation.py",
)


class AuditError(RuntimeError):
    pass


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise AuditError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise AuditError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verify_hash(path: Path, expected: str) -> Mapping[str, Any]:
    _require(path.is_file(), f"missing evidence file: {path}")
    actual = _sha256(path)
    _require(actual == expected, f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": actual, "passed": True}


def _verify_recursive_ledger(root: Path) -> Mapping[str, Any]:
    ledger_path = root / "artifact_hashes.json"
    ledger = _read_json(ledger_path)
    artifacts = ledger.get("artifacts")
    _require(isinstance(artifacts, Mapping) and artifacts, f"invalid recursive ledger: {ledger_path}")
    for relative, expected in artifacts.items():
        _verify_hash(root / str(relative), str(expected))
    return {
        "root": str(root.relative_to(REPO_ROOT)),
        "artifact_count": len(artifacts),
        "ledger_sha256": _sha256(ledger_path),
        "passed": True,
    }


def _registry_check() -> Mapping[str, Any]:
    registry = _read_json(P0_REGISTRY)
    rows = registry.get("cells")
    _require(isinstance(rows, list), "P0 registry cells are missing")
    ids = [str(row.get("cell_id")) for row in rows if isinstance(row, Mapping)]
    _require(len(ids) == len(set(ids)), "P0 registry contains duplicate cell IDs")
    _require(set(ids) == set(EXPECTED_STATES), "terminal matrix differs from P0 registry membership")
    return {
        "registry_path": str(P0_REGISTRY.relative_to(REPO_ROOT)),
        "registry_sha256": _sha256(P0_REGISTRY),
        "cell_ids": sorted(ids),
        "passed": True,
    }


def _terminal_evidence_checks() -> list[Mapping[str, Any]]:
    rows = []
    for cell_id in sorted(EXPECTED_STATES):
        relative, expected = EVIDENCE[cell_id]
        reference = _verify_hash(REPO_ROOT / relative, expected)
        rows.append(
            {
                "cell_id": cell_id,
                "terminal_state": EXPECTED_STATES[cell_id],
                "evidence": reference,
            }
        )
    return rows


def _full_target_signature(result: Mapping[str, Any]) -> str:
    identity = result.get("target_identity")
    _require(isinstance(identity, Mapping), "missing target identity")
    signature = identity.get("target_signature")
    _require(isinstance(signature, str) and len(signature) == 64, "missing full target signature")
    return signature


def _verify_run_manifest(
    path: Path, target_signature: str, *, allow_legacy_target_omission: bool = False
) -> Mapping[str, Any]:
    manifest = _read_json(path)
    required = (
        "git_commit",
        "command",
        "python_executable",
        "gpu_memory_policy",
        "jit_compile",
        "dtype",
        "tf32_execution_enabled",
        "wall_time_seconds",
        "output_root",
        "plan_file",
        "result_file",
        "nonclaims",
    )
    missing = [field for field in required if field not in manifest]
    _require(not missing, f"manifest missing fields {missing}: {path}")
    legacy_omissions = []
    if "target_signature" not in manifest:
        _require(allow_legacy_target_omission, f"manifest lacks target signature: {path}")
        legacy_omissions.append(
            "LEGACY_MANIFEST_TARGET_SIGNATURE_ABSENT_BOUND_BY_HASHED_RESULT"
        )
    seed_present = any(field in manifest for field in ("seeds", "random_seed", "random_seeds"))
    _require(seed_present, f"manifest lacks seed provenance: {path}")
    memory = manifest.get("gpu_memory_policy")
    _require(isinstance(memory, Mapping), f"invalid memory policy: {path}")
    _require(memory.get("all_physical_devices_memory_growth") is True, f"memory growth absent: {path}")
    _require(memory.get("configured_before_logical_device_initialization") is True, f"late memory policy: {path}")
    _require(memory.get("full_device_preallocation_disabled") is True, f"full preallocation enabled: {path}")
    _require(manifest.get("jit_compile") is True, f"XLA disabled: {path}")
    _require(manifest.get("tf32_execution_enabled") is True, f"TF32 provenance absent: {path}")
    if "target_signature" in manifest:
        _require(manifest.get("target_signature") == target_signature, f"manifest target mismatch: {path}")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "git_commit": manifest["git_commit"],
        "wall_time_seconds": manifest["wall_time_seconds"],
        "legacy_omissions": legacy_omissions,
        "passed": True,
    }


def _verify_final_diagnostics(result: Mapping[str, Any], *, role: str) -> Mapping[str, Any]:
    final = result.get("final_joint_diagnostic")
    _require(isinstance(final, Mapping), "missing final joint diagnostic")
    convergence = final.get("convergence")
    agreement = final.get("physical_mean_agreement")
    _require(isinstance(convergence, Mapping), "missing convergence diagnostic")
    _require(isinstance(agreement, Mapping), "missing physical-mean agreement")
    definitions = convergence.get("definitions")
    _require(isinstance(definitions, Mapping), "missing convergence definitions")
    _require(
        definitions.get("rhat") == "max(rank-normalized split R-hat, folded rank-normalized split R-hat)",
        "wrong modern R-hat definition",
    )
    _require(convergence.get("passed") is True, "final convergence failed")
    _require(float(convergence.get("max_rhat", float("inf"))) <= 1.01, "final R-hat failed")
    _require(float(convergence.get("min_bulk_ess", 0.0)) >= 1000.0, "final bulk ESS failed")
    _require(float(convergence.get("min_tail_ess", 0.0)) >= 400.0, "final tail ESS failed")
    _require(agreement.get("passed") is True, "physical-mean agreement failed")
    _require(agreement.get("supported_disagreement") is False, "supported disagreement present")
    _require(agreement.get("unresolved_precision") is False, "agreement precision unresolved")
    _require(all(row.get("passed") is True for row in agreement.get("parameter_rows", [])), "mean row failed")
    sequential = result.get("sequential_run")
    _require(isinstance(sequential, Mapping), "missing sequential run")
    _require(sequential.get("warmup_passed") is True, "warm-up failed")
    _require(sequential.get("warmup_excluded_from_posterior") is True, "warm-up pooled into inference")
    _require(int(sequential.get("retained_results_per_chain", 0)) <= 10000, "retained cap exceeded")
    _require(not sequential.get("hard_vetoes"), "final sampler hard veto present")
    return {
        "role": role,
        "draws_per_chain": convergence["draw_count_per_chain"],
        "max_rhat": convergence["max_rhat"],
        "min_bulk_ess": convergence["min_bulk_ess"],
        "min_tail_ess": convergence["min_tail_ess"],
        "agreement_estimand_count": len(agreement.get("parameter_rows", [])),
        "passed": True,
    }


def _verify_confirmation_cell(cell_id: str, root: Path) -> Mapping[str, Any]:
    result_path = root / "result.json"
    result = _read_json(result_path)
    _require(result.get("cell_id") == cell_id, f"confirmation cell mismatch: {cell_id}")
    _require(result.get("passed") is True and result.get("decision") == "NEUTRA_CONFIRMED", f"raw R4 result failed: {cell_id}")
    target_signature = _full_target_signature(result)
    recursive = [_verify_recursive_ledger(root)]

    training_reference = result.get("training_reference")
    comparator_reference = result.get("comparator_reference")
    _require(isinstance(training_reference, Mapping), f"missing training reference: {cell_id}")
    _require(isinstance(comparator_reference, Mapping), f"missing comparator reference: {cell_id}")
    training_root = REPO_ROOT / str(training_reference["root"])
    comparator_root = REPO_ROOT / str(comparator_reference["root"])
    recursive.extend((_verify_recursive_ledger(training_root), _verify_recursive_ledger(comparator_root)))
    _verify_hash(training_root / "result.json", str(training_reference["result_sha256"]))
    _verify_hash(comparator_root / "result.json", str(comparator_reference["result_sha256"]))

    training = _read_json(training_root / "result.json")
    comparator = _read_json(comparator_root / "result.json")
    retained_metadata = _read_json(root / "samples/retained/cumulative/metadata.json")
    binding = result.get("training_result_binding")
    _require(isinstance(binding, Mapping), f"missing training binding: {cell_id}")
    _require(training.get("target_identity", {}).get("target_signature") == target_signature, f"training target mismatch: {cell_id}")
    _require(comparator.get("target_identity", {}).get("target_signature") == target_signature, f"comparator target mismatch: {cell_id}")
    _require(retained_metadata.get("target_signature") == target_signature, f"retained target mismatch: {cell_id}")
    _require(retained_metadata.get("stage") == "retained", f"wrong retained archive stage: {cell_id}")
    _require(retained_metadata.get("warmup_excluded_from_posterior") is True, f"warm-up exclusion missing: {cell_id}")
    _require(training.get("steps") == 5000 and training.get("passed") is True, f"invalid final training: {cell_id}")
    _require(training.get("screen_weights_reused_by_final") is False, f"screen weights reused: {cell_id}")
    _require(training.get("frozen_trainable_parity", {}).get("passed") is True, f"training parity failed: {cell_id}")
    _require(training.get("transport_hash") == binding.get("transport_hash"), f"transport hash mismatch: {cell_id}")
    _require(training.get("transport_artifact_signature") == binding.get("artifact_signature"), f"artifact signature mismatch: {cell_id}")
    _require(training.get("training_state_hash") == binding.get("training_state_hash"), f"training state mismatch: {cell_id}")
    _require(comparator.get("passed") is True, f"comparator failed: {cell_id}")

    manifests = [
        _verify_run_manifest(root / "run_manifest.json", target_signature),
        _verify_run_manifest(
            training_root / "run_manifest.json",
            target_signature,
            allow_legacy_target_omission=True,
        ),
        _verify_run_manifest(comparator_root / "run_manifest.json", target_signature),
    ]
    final_diagnostics = _verify_final_diagnostics(result, role="terminal_confirmation")

    selected = result.get("selected_probe")
    _require(isinstance(selected, Mapping), f"missing selected probe: {cell_id}")
    verification = selected.get("tuning_verification")
    _require(isinstance(verification, Mapping), f"confirmation lacks disjoint tuning verifier: {cell_id}")
    modern = verification.get("modern_rhat")
    health = verification.get("health")
    archive = verification.get("archive")
    _require(verification.get("admitted") is True, f"tuning verifier not admitted: {cell_id}")
    _require(isinstance(modern, Mapping) and modern.get("passed") is True, f"tuning R-hat failed: {cell_id}")
    _require(float(modern.get("max_finite_rhat", float("inf"))) <= 1.01, f"tuning threshold failed: {cell_id}")
    _require(isinstance(health, Mapping) and health.get("health_passed") is True, f"tuning health failed: {cell_id}")
    _require(
        health.get("native_divergence_status")
        != "available"
        or int(health.get("native_divergence_count", 1)) == 0,
        f"positive native divergence: {cell_id}",
    )
    _require(isinstance(archive, Mapping), f"tuning archive missing: {cell_id}")
    _require(archive.get("target_signature") == target_signature, f"tuning archive target mismatch: {cell_id}")
    _require(archive.get("excluded_from_posterior") is True, f"tuning archive used for inference: {cell_id}")
    _require(tuple(archive.get("sample_shape", ())) == (1000, 4, int(modern.get("parameter_count", 0))), f"tuning archive shape mismatch: {cell_id}")
    _require(tuple(archive.get("seed", ())) == tuple(verification.get("seed", ())), f"tuning archive seed mismatch: {cell_id}")
    _require(int(archive.get("grid_index", -1)) == int(selected.get("grid_index", -2)), f"tuning archive grid mismatch: {cell_id}")
    tuning: Mapping[str, Any] = {
        "classification": "TUNING_ADMITTED",
        "disjoint_verifier": True,
        "modern_rhat": modern["max_finite_rhat"],
        "archive_excluded_from_posterior": True,
        "passed": True,
    }
    return {
        "cell_id": cell_id,
        "target_signature": target_signature,
        "raw_r4_result_sha256": _sha256(result_path),
        "recursive_ledgers": recursive,
        "manifests": manifests,
        "training": {
            "steps": training["steps"],
            "seed": training.get("seed"),
            "recipe_id": training.get("recipe_id"),
            "transport_hash": training.get("transport_hash"),
            "parity_passed": True,
        },
        "tuning_admission": tuning,
        "final_sampler_diagnostics": final_diagnostics,
    }


def _blocked_semantics() -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    p2_path = REPO_ROOT / EVIDENCE["SVX-SGQF"][0]
    p2 = _read_json(p2_path)
    selection = p2.get("selection_rows")
    thresholds = p2.get("thresholds")
    _require(isinstance(selection, list) and selection, "SVX selection ladder missing")
    _require(isinstance(thresholds, Mapping), "SVX thresholds missing")
    _require(all(row.get("passed") is False for row in selection), "SVX level unexpectedly passed")
    _require(
        float(p2["reference_level"]["prefix_dense_value_gap_per_observation"])
        > float(thresholds["prefix_dense_value_gap_per_observation"]),
        "SVX filter value blocker unsupported",
    )
    _require(p2.get("zhao_cui_cell_status") == "TARGET_BLOCKED_SOURCE_ROUTE_MISMATCH", "SVX-ZC source blocker unsupported")
    rows.extend(
        (
            {"cell_id": "SVX-SGQF", "semantic_basis": "no frozen SGQF level passed filter admission", "passed": True},
            {"cell_id": "SVX-ZC", "semantic_basis": "source-route mismatch and no production fixed route", "passed": True},
        )
    )

    p3 = _read_json(REPO_ROOT / EVIDENCE["KSC-UKF"][0])
    _require(p3.get("dense_reference", {}).get("passed") is True, "KSC dense reference invalid")
    admission = p3.get("filter_admission")
    thresholds = p3.get("thresholds")
    _require(isinstance(admission, Mapping) and admission.get("passed") is False, "KSC filter blocker absent")
    _require(float(admission["ukf_dense_score_gap"]) > float(thresholds["ukf_dense_score_gap"]), "KSC score blocker unsupported")
    rows.append({"cell_id": "KSC-UKF", "semantic_basis": "valid dense reference; UKF value/score margins failed", "passed": True})

    p4_ledger = _read_json(REPO_ROOT / EVIDENCE["PP-ZC"][0])
    _require(p4_ledger.get("states", {}).get("PP-ZC") == "TARGET_BLOCKED_SOURCE_ROUTE_MISMATCH", "PP-ZC blocker unsupported")
    rows.append({"cell_id": "PP-ZC", "semantic_basis": "production-ineligible extension/invention route", "passed": True})

    source_root = ARTIFACT_ROOT / "phase-p5/STR-UKF/plain-hmc/attempt-02"
    source = _read_json(source_root / "result.json")
    sequential = source.get("sequential_run")
    _require(isinstance(sequential, Mapping), "STR source sequential evidence missing")
    health = sequential.get("warmup_checks", [{}])[0].get("health", {})
    _require(
        int(health.get("energy_error_divergence_count", 0)) == 1,
        "STR historical extreme-log-accept marker drifted",
    )
    _require(int(sequential.get("retained_results_per_chain", -1)) == 0, "STR source retained evidence unexpectedly exists")
    affine = _read_json(REPO_ROOT / EVIDENCE["STR-UKF"][0])
    _require(affine.get("passed") is False and affine.get("score_gate_passed") is False, "STR affine mode blocker unsupported")
    _require(float(affine.get("final_score_norm_inf", 0.0)) > 1.0e-4, "STR terminal score blocker unsupported")
    rows.append(
        {
            "cell_id": "STR-UKF",
            "semantic_basis": (
                "affine terminal-score failure; historical source energy-veto "
                "interpretation retired because it counted one finite extreme "
                "log-acceptance proposal"
            ),
            "passed": True,
        }
    )

    structural_text = (REPO_ROOT / EVIDENCE["STR-ZC"][0]).read_text(encoding="utf-8")
    _require("TARGET_BLOCKED_EXTENSION_ROUTE_NOT_DESIGNED" in structural_text, "STR-ZC extension blocker unsupported")
    rows.append({"cell_id": "STR-ZC", "semantic_basis": "extension/invention target not designed", "passed": True})

    gpu = _read_json(REPO_ROOT / EVIDENCE["SIR-UKF"][0])
    gap = float(gpu["cells"]["SIR-UKF"]["score_cpu_parity"]["maximum_scale_normalized_gap"])
    _require(gap > 1.0e-7 and gpu["cells"]["SIR-UKF"]["passed"] is False, "SIR-UKF parity blocker unsupported")
    rows.append(
        {
            "cell_id": "SIR-UKF",
            "semantic_basis": "GPU/CPU score parity exceeded prospective limit",
            "observed_gap": gap,
            "limit": 1.0e-7,
            "passed": True,
        }
    )

    sir_zc_text = (REPO_ROOT / EVIDENCE["SIR-ZC"][0]).read_text(encoding="utf-8")
    for token in (
        "fixed SIR parameters",
        "three-parameter target may be studied as a BayesFilter extension",
        "Current fixed-TTSIRT substrate is HMC-score ready",
        "No Zhao-Cui observed-data parameter posterior, full retained-marginal score",
    ):
        _require(token in sir_zc_text, f"SIR-ZC source/derivative token missing: {token}")
    _require("unsupported and currently blocked" in sir_zc_text, "SIR-ZC blocked verdict missing")
    rows.append({"cell_id": "SIR-ZC", "semantic_basis": "parameter extension lacks observed-data retained-object score closure", "passed": True})
    return rows


def _static_policy_scan() -> Mapping[str, Any]:
    forbidden_hits = []
    loops = []
    for relative in ACTIVE_STATIC_PATHS:
        path = REPO_ROOT / relative
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                module = getattr(node, "module", None)
                if any(name == "numpy" or name.startswith("numpy.") for name in names) or module == "numpy":
                    forbidden_hits.append({"path": relative, "line": node.lineno, "kind": "numpy_import"})
            if isinstance(node, ast.Call):
                text = ast.get_source_segment(source, node.func) or ""
                if text in ("tf.numpy_function", "tf.py_function"):
                    forbidden_hits.append({"path": relative, "line": node.lineno, "kind": text})
            if isinstance(node, (ast.For, ast.AsyncFor)):
                target = ast.get_source_segment(source, node.target) or ""
                iterator = ast.get_source_segment(source, node.iter) or ""
                loops.append({"path": relative, "line": node.lineno, "target": target, "iterator": iterator[:160]})
    _require(not forbidden_hits, f"forbidden active-route static hits: {forbidden_hits}")
    suspicious = [
        row
        for row in loops
        if any(
            token in row["iterator"].lower()
            for token in ("samples", "sample_rows", "draws", "draw_rows", "batch_rows")
        )
    ]
    _require(not suspicious, f"possible Python sample-axis loops: {suspicious}")
    return {
        "paths": list(ACTIVE_STATIC_PATHS),
        "numpy_or_host_callback_hits": forbidden_hits,
        "python_loops": loops,
        "loop_classification": "metadata, configuration, stage construction, or chunk orchestration; no sample-axis loop token found",
        "passed": True,
    }


def _inherited_claim_scan() -> Mapping[str, Any]:
    paths = (
        "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p2-exact-sv-result-2026-07-15.md",
        "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p3-ksc-ukf-result-2026-07-15.md",
        "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p4-predator-prey-result-2026-07-16.md",
        "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p5-structural-result-2026-07-16.md",
        "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p6-parameterized-sir-result-2026-07-16.md",
    )
    findings = []
    p4_text = (REPO_ROOT / paths[2]).read_text(encoding="utf-8")
    for token in (
        "d9b4f603b28acb06154ab554f41f745c5f544e2516ba4969c6b21d9e5268bacf",
        "a77d5edf2b8129d6ff95844e9c5d4bb94b7125c9997777b517f36b830fbda9c4",
        "fresh disjoint tuning verifier",
    ):
        _require(token in p4_text.lower(), f"P4 repair evidence missing from phase result: {token}")
    for relative in paths:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        _require("Default readiness" in text or "Default-readiness" in text, f"missing readiness classification: {relative}")
        _require("Statistically supported ranking" in text, f"missing ranking classification: {relative}")
    return {"paths": list(paths), "findings": findings, "passed_with_correction": True}


def _artifact_hashes(root: Path) -> Mapping[str, Any]:
    return {
        "schema": "bayesfilter.multimodel_neutra_p7_hashes.v1",
        "artifacts": {
            str(path.relative_to(root)): _sha256(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "artifact_hashes.json"
        },
    }


def run_audit(output_root: Path) -> Mapping[str, Any]:
    started = time.monotonic()
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "P7 must explicitly hide GPU devices")
    registry = _registry_check()
    terminal_evidence = _terminal_evidence_checks()
    r4_rows = [_verify_confirmation_cell(cell_id, root) for cell_id, root in R4_ROOTS.items()]
    target_signatures = [row["target_signature"] for row in r4_rows]
    _require(len(target_signatures) == len(set(target_signatures)), "confirmed/diagnostic R4 target signature collision")
    blockers = _blocked_semantics()
    static_scan = _static_policy_scan()
    claim_scan = _inherited_claim_scan()

    cell_ledger = {
        "schema": "bayesfilter.multimodel_neutra_p7_cell_ledger.v1",
        "program_id": PROGRAM_ID,
        "states": EXPECTED_STATES,
        "cells": [
            {
                "cell_id": cell_id,
                "terminal_state": EXPECTED_STATES[cell_id],
                "evidence_path": EVIDENCE[cell_id][0],
                "evidence_sha256": EVIDENCE[cell_id][1],
                "earliest_reentry_rung": {
                    "SVX-SGQF": "R1_FILTER_ADMISSION",
                    "SVX-ZC": "R0_SOURCE_ROUTE_DESIGN",
                    "KSC-UKF": "R1_FILTER_ADMISSION",
                    "PP-SGQF": "COMPLETE_AT_SIX_MEAN_SCOPE",
                    "PP-UKF": "COMPLETE_AT_SIX_MEAN_SCOPE",
                    "PP-ZC": "R0_SOURCE_ROUTE_DESIGN",
                    "STR-UKF": "R2_COMPARATOR_GEOMETRY",
                    "STR-ZC": "R0_EXTENSION_TARGET_DESIGN",
                    "SIR-SGQF": "COMPLETE_AT_THREE_MEAN_SCOPE",
                    "SIR-UKF": "R1_GPU_SCORE_PARITY",
                    "SIR-ZC": "R0_OBSERVED_DATA_SCORE_ROUTE",
                }[cell_id],
            }
            for cell_id in sorted(EXPECTED_STATES)
        ],
    }

    result = {
        "schema": "bayesfilter.multimodel_neutra_p7_audit.v1",
        "program_id": PROGRAM_ID,
        "completed": True,
        "passed": True,
        "decision": "CELL_COMPLETE_WITH_BLOCKERS",
        "terminal_counts": {
            "mandatory_cells": 11,
            "neutra_confirmed": 3,
            "tuning_admission_evidence_blocked": 0,
            "other_precise_blockers": 8,
        },
        "registry_audit": registry,
        "terminal_evidence_audit": terminal_evidence,
        "r4_evidence_audit": r4_rows,
        "blocked_state_semantics": blockers,
        "static_policy_scan": static_scan,
        "claim_scan": claim_scan,
        "corrections": [
            {
                "cells": ["PP-UKF", "PP-SGQF"],
                "from": "PROVISIONAL_EVIDENCE_BLOCKED_TUNING_ADMISSION",
                "to": "NEUTRA_CONFIRMED",
                "reason": "fresh disjoint modern-R-hat <=1.01 tuning verification followed by fresh passing warm-up and retained confirmation",
                "historical_attempt_01_preserved": True,
            }
        ],
        "decision_table": {
            "decision": "close program with precise blockers",
            "primary_criterion_status": True,
            "veto_status": "clear for three narrow confirmations; eight cells retain precise local blockers",
            "main_uncertainty": "single-fixture mean-level evidence for the three confirmed cells",
            "next_justified_action": "write terminal result and reset memo; optional repairs start at named rungs",
            "not_concluded": [
                "no universal NeuTra success",
                "no full-distribution equivalence",
                "no filter exactness, ranking, calibration, robustness, production, or default readiness",
            ],
        },
        "inference_status": {
            "hard_veto_screen": "three narrow confirmations valid; eight cells precisely blocked",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": "runtime, acceptance, loss, quantile, SD, correlation, and aggregate counts",
            "default_readiness": False,
            "next_evidence_needed": "cell-specific repairs at earliest invalid rung plus new prospective confirmation where required",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_write_json(output_root / "result.json", result)
    _atomic_write_json(output_root / "cell_ledger.json", cell_ledger)
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "schema": "bayesfilter.multimodel_neutra_p7_manifest.v1",
        "program_id": PROGRAM_ID,
        "git_commit": commit,
        "dirty_worktree_disclosure": "shared dirty worktree; scoped read-only audit plus fresh P7 artifacts",
        "command": " ".join(sys.argv),
        "python_executable": sys.executable,
        "environment": "tf-gpu conda environment; standard-library-only P7 audit",
        "cpu_gpu_status": "CPU-only; CUDA_VISIBLE_DEVICES=-1 before process start",
        "numerical_framework_imported": False,
        "wall_time_seconds": time.monotonic() - started,
        "output_root": str(output_root.relative_to(REPO_ROOT)),
        "plan_file": str(PLAN_PATH.relative_to(REPO_ROOT)),
        "result_file": str((output_root / "result.json").relative_to(REPO_ROOT)),
        "random_seeds": "N/A: deterministic artifact integrity audit",
        "data_version": "P0 registry and P2-P6 preserved artifacts",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "nonclaims": result["decision_table"]["not_concluded"],
    }
    _atomic_write_json(output_root / "run_manifest.json", manifest)
    _atomic_write_json(output_root / "artifact_hashes.json", _artifact_hashes(output_root))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    result = run_audit(output_root)
    print(json.dumps({"decision": result["decision"], "passed": result["passed"], "terminal_counts": result["terminal_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
