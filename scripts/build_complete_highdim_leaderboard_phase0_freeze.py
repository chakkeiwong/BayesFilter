#!/usr/bin/env python3
"""Build the source-bound Phase 0 complete-leaderboard freeze artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "docs/plans/artifacts/complete-highdim-leaderboard/"
    "phase0-boundary-freeze-2026-07-11.json"
)
SCHEMA_VERSION = "bayesfilter.complete_highdim_leaderboard.phase0_freeze.v2"
ORIGINAL_FREEZE_SHA256 = (
    "4115ef55114ffd73255363f0c62c4a19dd85d7ca3241d002c48409cb9004f878"
)
AUTHORITY_AMENDMENT_PATH = (
    "docs/plans/"
    "bayesfilter-complete-highdim-leaderboard-phase0-owner-authority-amendment-2026-07-12.md"
)
AUTHORITY_AMENDMENT_SHA256 = (
    "171b8d42ec5f31869181003636a223b4e3063e38c03a66e1df7f77782d81ce90"
)
SIR_TARGET_GENERATION_IDENTITY = (
    "fixed_bayesfilter_sir_observations_from_dataset_seed_81103_"
    "not_author_matlab_rng1_reproduction"
)

ALGORITHMS = (
    "fixed_sgqf",
    "ukf",
    "zhao_cui_scalar_or_multistate",
    "ledh_pfpf_ot",
)
MAIN_ROWS = (
    "benchmark_lgssm_exact_oracle_m3_T50",
    "zhao_cui_sv_actual_nongaussian_T1000",
    "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
    "zhao_cui_spatial_sir_austria_j9_T20",
    "zhao_cui_predator_prey_T20",
    "zhao_cui_generalized_sv_synthetic_from_estimated_values",
)
SIDECAR_ROW = "zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale"
FROZEN_SEEDS = (81120, 81121, 81122, 81123, 81124)

NONLEDH_BASELINE_PATH = (
    "docs/plans/bayesfilter-two-lane-highdim-leaderboard-results-2026-07-03.json"
)
HISTORICAL_LEADERBOARD_PATH = (
    "docs/plans/bayesfilter-two-lane-highdim-ledh-inclusive-leaderboard-results-2026-07-06.json"
)

EXPECTED_INPUT_HASHES = {
    AUTHORITY_AMENDMENT_PATH: AUTHORITY_AMENDMENT_SHA256,
    NONLEDH_BASELINE_PATH: "b44fd1ccc8a0132d45ea4f64925bd92930a17c11f7b62bc8f0a15f66631985e7",
    HISTORICAL_LEADERBOARD_PATH: "57317fb8f0b4a55c3357a7014f1d68647278657b11843460f90e4f95383900d0",
    "docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json": (
        "21e87489c8eb661db4b2e9b27cefb4e45e567a8c0bb4743ffd4f09feec3faf93"
    ),
    "docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json": (
        "38a7da0ef1f32f96e74d4f62676d823af2fbe1b4267d88dbfa0c39c4156ba9b8"
    ),
    "docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json": (
        "17eaaf23302fa68e802eef686b167e4b31cc3dba755503f9b74343d2ca29ef45"
    ),
    "docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json": (
        "3811268078d07e0ac4c2fcd9400af156a5918503e404937d516391ce0f034c16"
    ),
    "docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json": (
        "5afb71144576bdb0070080f684b5d5b41f33de77889105b10bcd78e36b77dd77"
    ),
    "docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json": (
        "9883721faf8af9fbe96ef75c209f86eda5732aec6ca5e602980d4cf27338b3b6"
    ),
    "bayesfilter/highdim/ledh_score_contract.py": (
        "aa15f058b30850c940b978491080893353c519c3ee31a344d0d42f20b81aeef3"
    ),
    "bayesfilter/ledh_fd_policy.py": (
        "32c20ab5467c464a32bd2f098b0a1f1c0e67765890007126349abc6434edd2b5"
    ),
    "docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py": (
        "2bd7c4c62773657213ccd488c9e55b96f3f7d6d4a3b00a7aaf2a8fb070031d58"
    ),
    "docs/benchmarks/benchmark_two_lane_highdim_ledh_inclusive_results.py": (
        "dcc176e4e3533abfd609b27fc52db3dc3c608de27d88606e15f4ae8bb60bd365"
    ),
    "experiments/dpf_implementation/tf_tfp/filters/experimental_batched_ledh_pfpf_ot_tf.py": (
        "a9d680cc90ad59655a35268766213bb452d6ab703993918600148194364383fe"
    ),
    "docs/plans/bayesfilter-ledh-predator-generalized-fd-root-cause-repair-result-2026-07-11.md": (
        "42630b9ab97cdcb39d4ecd8c0fdc172647a63b86c5c3a478fd8efd23352f1fed"
    ),
}

ROW_SPECS = {
    "benchmark_lgssm_exact_oracle_m3_T50": {
        "source_value_artifact": "docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json",
        "time_steps": 50,
        "num_particles": 10000,
        "target_observation_policy": "lgssm_gaussian_observation_density",
        "theta_coordinate_system": "physical_benchmark_exact_oracle",
        "parameter_order": ["phi1", "phi2", "phi3", "q_scale", "r_scale"],
        "evaluation_theta": [0.72, 0.55, 0.35, 0.35, 0.45],
    },
    "zhao_cui_spatial_sir_austria_j9_T20": {
        "source_value_artifact": "docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json",
        "time_steps": 20,
        "num_particles": 10000,
        "target_observation_policy": "fixed_sir_infectious_components_gaussian_observation_density",
        "theta_coordinate_system": "sir_log_scale_theta",
        "parameter_order": ["log_kappa_scale", "log_nu_scale", "log_obs_noise_scale"],
        "evaluation_theta": [0.0, 0.0, 0.0],
    },
    "zhao_cui_predator_prey_T20": {
        "source_value_artifact": "docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json",
        "time_steps": 20,
        "num_particles": 10000,
        "target_observation_policy": "additive_gaussian_predator_prey",
        "theta_coordinate_system": "physical",
        "parameter_order": ["r", "K", "a", "s", "u", "v"],
        "evaluation_theta": [0.6, 114.0, 25.0, 0.3, 0.5, 0.5],
    },
    "zhao_cui_sv_actual_nongaussian_T1000": {
        "source_value_artifact": "docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json",
        "time_steps": 1000,
        "num_particles": 10000,
        "target_observation_policy": "transformed_actual_sv_log_y_square",
        "theta_coordinate_system": "synthetic_unconstrained",
        "parameter_order": ["gamma_unconstrained", "log_beta"],
        "evaluation_theta": [0.2533471031357997, -0.916290731874155],
    },
    "zhao_cui_generalized_sv_synthetic_from_estimated_values": {
        "source_value_artifact": "docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json",
        "time_steps": 1008,
        "num_particles": 10000,
        "target_observation_policy": "source_route_prior_mean_generalized_sv",
        "theta_coordinate_system": "source_route_active_transformed_prior_mean",
        "parameter_order": ["gamma_unconstrained", "log_tau", "mu"],
        "evaluation_theta": [1.0824113944610982, -2.076793740349318, 0.0],
    },
    "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000": {
        "source_value_artifact": "docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json",
        "time_steps": 1000,
        "num_particles": 10000,
        "target_observation_policy": "ksc_log_chi_square_gaussian_mixture_surrogate",
        "theta_coordinate_system": "synthetic_unconstrained",
        "parameter_order": ["gamma_unconstrained", "log_beta"],
        "evaluation_theta": [0.2533471031357997, -0.916290731874155],
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_sha_bound(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed = _sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"input SHA-256 mismatch for {path}: {observed} != {expected_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"input must be a JSON object: {path}")
    return payload


def _git_output(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _forward_row(spec: Mapping[str, Any]) -> dict[str, Any]:
    rel = str(spec["source_value_artifact"])
    payload = _load_sha_bound(ROOT / rel, EXPECTED_INPUT_HASHES[rel])
    contract = payload.get("forward_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"missing forward_contract: {rel}")
    theta = contract.get("theta_contract")
    if not isinstance(theta, Mapping):
        raise ValueError(f"missing theta_contract: {rel}")
    observed = {
        "source_value_artifact": rel,
        "source_value_artifact_sha256": EXPECTED_INPUT_HASHES[rel],
        "row_id": payload.get("row_id"),
        "row_scope": contract.get("row_scope"),
        "full_leaderboard_row": contract.get("full_leaderboard_row"),
        "time_steps": payload.get("time_steps"),
        "num_particles": payload.get("num_particles"),
        "batch_seeds": payload.get("batch_seeds"),
        "target_observation_policy": payload.get("target_observation_policy"),
        "theta_coordinate_system": payload.get("theta_coordinate_system"),
        "parameter_order": theta.get("parameter_order"),
        "evaluation_theta": theta.get("truth_theta"),
    }
    expected = {
        **dict(spec),
        "source_value_artifact_sha256": EXPECTED_INPUT_HASHES[rel],
        "row_id": payload.get("row_id"),
        "row_scope": "main_observed_data_filtering_row",
        "full_leaderboard_row": True,
        "batch_seeds": list(FROZEN_SEEDS),
    }
    if observed != expected:
        raise ValueError(f"forward row metadata mismatch for {rel}: {observed!r}")
    return observed


def _cell_status(
    row: Mapping[str, Any],
    *,
    source_artifact: str,
) -> dict[str, Any]:
    row_id = str(row["row_id"])
    algorithm = str(row["algorithm_id"])
    executed_value_score = (
        row.get("comparison_status") == "executed_value_score"
        and row.get("average_log_likelihood") is not None
        and isinstance(row.get("score"), Sequence)
        and not isinstance(row.get("score"), (str, bytes))
    )
    if row_id == SIDECAR_ROW:
        closure = {
            "fixed_sgqf": "not_applicable_scoped_sidecar",
            "ukf": "not_applicable_scoped_sidecar",
            "zhao_cui_scalar_or_multistate": (
                "historical_scoped_component_candidate_outside_program"
            ),
            "ledh_pfpf_ot": "historical_scoped_diagnostic_outside_program",
        }[algorithm]
    elif algorithm == "ledh_pfpf_ot":
        closure = "gap_current_source_five_seed_ledh_admission"
    elif row_id in MAIN_ROWS[:3] and executed_value_score:
        closure = "frozen_nonledh_baseline_candidate"
    else:
        closure = "gap_target_matched_value_and_score_evaluator"
    return {
        "row_id": row_id,
        "algorithm_id": algorithm,
        "status_source_artifact": source_artifact,
        "status_source_artifact_sha256": EXPECTED_INPUT_HASHES[source_artifact],
        "historical_comparison_status": row.get("comparison_status"),
        "historical_score_status": row.get("score_status"),
        "historical_value_present": row.get("average_log_likelihood") is not None,
        "historical_score_present": row.get("score") is not None,
        "phase0_closure_status": closure,
        "current_program_admitted": False,
    }


def validate_freeze(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("wrong Phase 0 schema version")
    if tuple(payload.get("main_rows") or ()) != MAIN_ROWS:
        raise ValueError("main-row freeze mismatch")
    if tuple(payload.get("algorithms") or ()) != ALGORITHMS:
        raise ValueError("algorithm freeze mismatch")
    supersession = payload.get("authority_supersession")
    expected_supersession = {
        "authority_amendment_path": AUTHORITY_AMENDMENT_PATH,
        "authority_amendment_sha256": AUTHORITY_AMENDMENT_SHA256,
        "original_phase0_freeze_sha256": ORIGINAL_FREEZE_SHA256,
        "supersession_scope": "sir_target_generation_identity_and_exact_row_extension_classifications_only",
    }
    if supersession != expected_supersession:
        raise ValueError("Phase 0 authority-supersession binding mismatch")
    sidecar = payload.get("sidecar")
    if not isinstance(sidecar, Mapping) or sidecar.get("row_id") != SIDECAR_ROW:
        raise ValueError("parameterized-SIR sidecar freeze mismatch")
    if SIDECAR_ROW in tuple(payload.get("main_rows") or ()):
        raise ValueError("parameterized-SIR sidecar cannot be a main row")
    cells = payload.get("starting_cells")
    if not isinstance(cells, list) or len(cells) != 24:
        raise ValueError("Phase 0 must freeze exactly 24 main cells")
    keys = [(cell.get("row_id"), cell.get("algorithm_id")) for cell in cells]
    expected_keys = [(row, algorithm) for row in MAIN_ROWS for algorithm in ALGORITHMS]
    if keys != expected_keys or len(set(keys)) != 24:
        raise ValueError("Phase 0 main-cell matrix mismatch")
    frozen = sum(
        cell.get("phase0_closure_status") == "frozen_nonledh_baseline_candidate"
        for cell in cells
    )
    gaps = sum(str(cell.get("phase0_closure_status", "")).startswith("gap_") for cell in cells)
    summary = payload.get("summary")
    expected_summary = {
        "num_main_rows": 6,
        "num_algorithms": 4,
        "num_main_cells": 24,
        "num_sidecar_rows": 1,
        "num_frozen_nonledh_baseline_candidates": 9,
        "num_current_closure_gaps": 15,
        "num_current_program_admitted_cells": 0,
        "num_current_source_ledh_five_seed_aggregates": 0,
        "numeric_leaderboard_complete": False,
    }
    if frozen != 9 or gaps != 15 or summary != expected_summary:
        raise ValueError("Phase 0 starting-state summary mismatch")
    for cell in cells:
        algorithm = cell.get("algorithm_id")
        expected_source = (
            HISTORICAL_LEADERBOARD_PATH
            if algorithm == "ledh_pfpf_ot"
            else NONLEDH_BASELINE_PATH
        )
        if (
            cell.get("status_source_artifact") != expected_source
            or cell.get("status_source_artifact_sha256")
            != EXPECTED_INPUT_HASHES[expected_source]
        ):
            raise ValueError("Phase 0 cell source-binding mismatch")
    sidecar_cells = sidecar.get("cells")
    if not isinstance(sidecar_cells, list) or len(sidecar_cells) != 4:
        raise ValueError("Phase 0 sidecar must contain exactly four cells")
    for cell in sidecar_cells:
        algorithm = cell.get("algorithm_id")
        expected_source = (
            HISTORICAL_LEADERBOARD_PATH
            if algorithm == "ledh_pfpf_ot"
            else NONLEDH_BASELINE_PATH
        )
        expected_closure = {
            "fixed_sgqf": "not_applicable_scoped_sidecar",
            "ukf": "not_applicable_scoped_sidecar",
            "zhao_cui_scalar_or_multistate": (
                "historical_scoped_component_candidate_outside_program"
            ),
            "ledh_pfpf_ot": "historical_scoped_diagnostic_outside_program",
        }.get(algorithm)
        if (
            cell.get("status_source_artifact") != expected_source
            or cell.get("status_source_artifact_sha256")
            != EXPECTED_INPUT_HASHES[expected_source]
            or cell.get("phase0_closure_status") != expected_closure
            or cell.get("current_program_admitted") is not False
        ):
            raise ValueError("Phase 0 sidecar source-binding mismatch")
    forward = payload.get("ledh_forward_rows")
    if not isinstance(forward, list) or {entry.get("row_id") for entry in forward} != set(MAIN_ROWS):
        raise ValueError("Phase 0 LEDH forward-row set mismatch")


def build_freeze() -> dict[str, Any]:
    for rel, digest in EXPECTED_INPUT_HASHES.items():
        path = ROOT / rel
        if not path.is_file() or _sha256(path) != digest:
            observed = None if not path.is_file() else _sha256(path)
            raise ValueError(f"frozen input mismatch for {rel}: {observed} != {digest}")

    baseline = _load_sha_bound(
        ROOT / NONLEDH_BASELINE_PATH,
        EXPECTED_INPUT_HASHES[NONLEDH_BASELINE_PATH],
    )
    baseline_rows = baseline.get("rows")
    if not isinstance(baseline_rows, list):
        raise ValueError("non-LEDH baseline rows must be a list")
    baseline_by_key = {
        (row.get("row_id"), row.get("algorithm_id")): row for row in baseline_rows
    }
    expected_baseline = {
        (row, algorithm)
        for row in (*MAIN_ROWS, SIDECAR_ROW)
        for algorithm in ALGORITHMS
        if algorithm != "ledh_pfpf_ot"
    }
    if set(baseline_by_key) != expected_baseline:
        raise ValueError("non-LEDH baseline does not contain the exact seven-row matrix")

    historical = _load_sha_bound(
        ROOT / HISTORICAL_LEADERBOARD_PATH,
        EXPECTED_INPUT_HASHES[HISTORICAL_LEADERBOARD_PATH],
    )
    rows = historical.get("rows")
    if not isinstance(rows, list):
        raise ValueError("historical leaderboard rows must be a list")
    by_key = {(row.get("row_id"), row.get("algorithm_id")): row for row in rows}
    expected_all = {
        (row, algorithm)
        for row in (*MAIN_ROWS, SIDECAR_ROW)
        for algorithm in ALGORITHMS
    }
    if set(by_key) != expected_all:
        raise ValueError("historical leaderboard does not contain the exact seven-row matrix")

    def source_row(row: str, algorithm: str) -> tuple[Mapping[str, Any], str]:
        if algorithm == "ledh_pfpf_ot":
            return by_key[(row, algorithm)], HISTORICAL_LEADERBOARD_PATH
        return baseline_by_key[(row, algorithm)], NONLEDH_BASELINE_PATH

    starting_cells = []
    for row in MAIN_ROWS:
        for algorithm in ALGORITHMS:
            source, source_artifact = source_row(row, algorithm)
            starting_cells.append(
                _cell_status(source, source_artifact=source_artifact)
            )
    sidecar_cells = []
    for algorithm in ALGORITHMS:
        source, source_artifact = source_row(SIDECAR_ROW, algorithm)
        sidecar_cells.append(_cell_status(source, source_artifact=source_artifact))
    ledh_rows = [_forward_row(ROW_SPECS[row]) for row in MAIN_ROWS]
    input_bindings = [
        {"path": rel, "sha256": digest, "role": _input_role(rel)}
        for rel, digest in EXPECTED_INPUT_HASHES.items()
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_status": "completed",
        "program_status": "phase0_boundary_freeze_only_not_admission",
        "authority_supersession": {
            "authority_amendment_path": AUTHORITY_AMENDMENT_PATH,
            "authority_amendment_sha256": AUTHORITY_AMENDMENT_SHA256,
            "original_phase0_freeze_sha256": ORIGINAL_FREEZE_SHA256,
            "supersession_scope": (
                "sir_target_generation_identity_and_exact_row_extension_"
                "classifications_only"
            ),
        },
        "main_rows": list(MAIN_ROWS),
        "algorithms": list(ALGORITHMS),
        "sidecar": {
            "row_id": SIDECAR_ROW,
            "row_scope": "scoped_component_row",
            "target_scope": "local_complete_data_zhao_cui_sir_d18_component",
            "time_steps": 20,
            "parameter_order": [
                "log_kappa_scale",
                "log_nu_scale",
                "log_obs_noise_scale",
            ],
            "closure_required_by_current_program": False,
            "target_signature_status": "outside_main_24_cell_program_not_frozen",
            "counted_as_main_row": False,
            "cells": sidecar_cells,
        },
        "ledh_forward_rows": ledh_rows,
        "starting_cells": starting_cells,
        "summary": {
            "num_main_rows": 6,
            "num_algorithms": 4,
            "num_main_cells": 24,
            "num_sidecar_rows": 1,
            "num_frozen_nonledh_baseline_candidates": 9,
            "num_current_closure_gaps": 15,
            "num_current_program_admitted_cells": 0,
            "num_current_source_ledh_five_seed_aggregates": 0,
            "numeric_leaderboard_complete": False,
        },
        "input_bindings": input_bindings,
        "policies": {
            "admitted_cell_value_scalar": "total_observed_data_log_likelihood",
            "admitted_cell_score_scalar": (
                "total_derivative_of_total_observed_data_log_likelihood"
            ),
            "average_log_likelihood_status": "derived_display_only_total_divided_by_time_steps",
            "canonical_target_signature_status": (
                "pending_phase1_byte_level_prepared_input_freeze"
            ),
            "ledh_execution": "trusted_gpu_xla_float32_tf32",
            "ledh_fd_rule": "max_coordinate_relative_error <= 0.05 * sqrt(num_parameters)",
            "ledh_seed_set": list(FROZEN_SEEDS),
            "ledh_seed_semantics": (
                "ordered_execution_seeds_for_main_row_value_score_fd_pairs_not_target_generation"
            ),
            "target_generation_identities": {
                "benchmark_lgssm_exact_oracle_m3_T50": "dataset_seed_81100",
                "zhao_cui_sv_actual_nongaussian_T1000": "dataset_seed_81101",
                "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000": (
                    "dataset_seed_81101_distinct_ksc_target_density"
                ),
                "zhao_cui_spatial_sir_austria_j9_T20": (
                    SIR_TARGET_GENERATION_IDENTITY
                ),
                "zhao_cui_predator_prey_T20": "dataset_seed_81104",
                "zhao_cui_generalized_sv_synthetic_from_estimated_values": (
                    "dataset_seed_81105"
                ),
            },
            "zhao_cui_route": "fixed_variant_source_route_paper_and_author_source_anchors_required",
            "retained_grid_route": "diagnostic_only_not_production_admissible",
            "runtime_cross_ranking_allowed": False,
        },
        "run_manifest": {
            "git_commit": _git_output("git", "rev-parse", "HEAD"),
            "working_directory": str(ROOT),
            "generator": "scripts/build_complete_highdim_leaderboard_phase0_freeze.py",
            "command": (
                "python scripts/build_complete_highdim_leaderboard_phase0_freeze.py "
                "--output docs/plans/artifacts/complete-highdim-leaderboard/"
                "phase0-boundary-freeze-2026-07-11.json"
            ),
            "random_seeds": "N/A; metadata freeze only",
            "cpu_gpu_status": "N/A; no framework or device initialization",
        },
        "nonclaims": [
            "Phase 0 does not admit any numerical leaderboard cell",
            "historical executed status is not current-source LEDH admission",
            "July 7 LEDH values are historical target/shape evidence only",
            "frozen non-LEDH candidates still require final row-target and score validation",
            "not HMC readiness, posterior correctness, ranking, superiority, or confidence coverage",
        ],
    }
    validate_freeze(payload)
    return payload


def _input_role(rel: str) -> str:
    if rel == NONLEDH_BASELINE_PATH:
        return "frozen_nonledh_baseline_candidate_source"
    if rel == AUTHORITY_AMENDMENT_PATH:
        return "owner_authority_amendment"
    if rel == HISTORICAL_LEADERBOARD_PATH:
        return "historical_status_only_not_current_admission"
    if "forward-scalar-artifact" in rel:
        return "frozen_ledh_target_shape_historical_forward_only"
    if rel.endswith("root-cause-repair-result-2026-07-11.md"):
        return "current_fd_and_manual_jvp_repair_authority"
    return "current_source_identity"


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    expected = build_freeze()
    if args.check:
        observed = json.loads(output.read_text(encoding="utf-8"))
        validate_freeze(observed)
        if observed != expected:
            raise ValueError("stored Phase 0 freeze does not match current frozen inputs")
        print(f"PHASE0_FREEZE_CHECK_PASS {output}")
        return 0
    _write(output, expected)
    print(f"PHASE0_FREEZE_WRITTEN {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
