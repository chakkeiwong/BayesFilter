#!/usr/bin/env python3
"""Independently audit the stored complete-leaderboard Phase 0 freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
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
SEEDS = (81120, 81121, 81122, 81123, 81124)

NONLEDH_SOURCE = (
    "docs/plans/bayesfilter-two-lane-highdim-leaderboard-results-2026-07-03.json"
)
LEDH_HISTORY_SOURCE = (
    "docs/plans/bayesfilter-two-lane-highdim-ledh-inclusive-leaderboard-results-2026-07-06.json"
)

EXPECTED_INPUTS = {
    AUTHORITY_AMENDMENT_PATH: (
        AUTHORITY_AMENDMENT_SHA256,
        "owner_authority_amendment",
    ),
    NONLEDH_SOURCE: (
        "b44fd1ccc8a0132d45ea4f64925bd92930a17c11f7b62bc8f0a15f66631985e7",
        "frozen_nonledh_baseline_candidate_source",
    ),
    LEDH_HISTORY_SOURCE: (
        "57317fb8f0b4a55c3357a7014f1d68647278657b11843460f90e4f95383900d0",
        "historical_status_only_not_current_admission",
    ),
    "docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json": (
        "21e87489c8eb661db4b2e9b27cefb4e45e567a8c0bb4743ffd4f09feec3faf93",
        "frozen_ledh_target_shape_historical_forward_only",
    ),
    "docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json": (
        "38a7da0ef1f32f96e74d4f62676d823af2fbe1b4267d88dbfa0c39c4156ba9b8",
        "frozen_ledh_target_shape_historical_forward_only",
    ),
    "docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json": (
        "17eaaf23302fa68e802eef686b167e4b31cc3dba755503f9b74343d2ca29ef45",
        "frozen_ledh_target_shape_historical_forward_only",
    ),
    "docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json": (
        "3811268078d07e0ac4c2fcd9400af156a5918503e404937d516391ce0f034c16",
        "frozen_ledh_target_shape_historical_forward_only",
    ),
    "docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json": (
        "5afb71144576bdb0070080f684b5d5b41f33de77889105b10bcd78e36b77dd77",
        "frozen_ledh_target_shape_historical_forward_only",
    ),
    "docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json": (
        "9883721faf8af9fbe96ef75c209f86eda5732aec6ca5e602980d4cf27338b3b6",
        "frozen_ledh_target_shape_historical_forward_only",
    ),
    "bayesfilter/highdim/ledh_score_contract.py": (
        "aa15f058b30850c940b978491080893353c519c3ee31a344d0d42f20b81aeef3",
        "current_source_identity",
    ),
    "bayesfilter/ledh_fd_policy.py": (
        "32c20ab5467c464a32bd2f098b0a1f1c0e67765890007126349abc6434edd2b5",
        "current_source_identity",
    ),
    "docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py": (
        "2bd7c4c62773657213ccd488c9e55b96f3f7d6d4a3b00a7aaf2a8fb070031d58",
        "current_source_identity",
    ),
    "docs/benchmarks/benchmark_two_lane_highdim_ledh_inclusive_results.py": (
        "dcc176e4e3533abfd609b27fc52db3dc3c608de27d88606e15f4ae8bb60bd365",
        "current_source_identity",
    ),
    "experiments/dpf_implementation/tf_tfp/filters/experimental_batched_ledh_pfpf_ot_tf.py": (
        "a9d680cc90ad59655a35268766213bb452d6ab703993918600148194364383fe",
        "current_source_identity",
    ),
    "docs/plans/bayesfilter-ledh-predator-generalized-fd-root-cause-repair-result-2026-07-11.md": (
        "42630b9ab97cdcb39d4ecd8c0fdc172647a63b86c5c3a478fd8efd23352f1fed",
        "current_fd_and_manual_jvp_repair_authority",
    ),
}

ROW_SPECS = {
    "benchmark_lgssm_exact_oracle_m3_T50": (
        50,
        "lgssm_gaussian_observation_density",
        "physical_benchmark_exact_oracle",
        ("phi1", "phi2", "phi3", "q_scale", "r_scale"),
        (0.72, 0.55, 0.35, 0.35, 0.45),
    ),
    "zhao_cui_sv_actual_nongaussian_T1000": (
        1000,
        "transformed_actual_sv_log_y_square",
        "synthetic_unconstrained",
        ("gamma_unconstrained", "log_beta"),
        (0.2533471031357997, -0.916290731874155),
    ),
    "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000": (
        1000,
        "ksc_log_chi_square_gaussian_mixture_surrogate",
        "synthetic_unconstrained",
        ("gamma_unconstrained", "log_beta"),
        (0.2533471031357997, -0.916290731874155),
    ),
    "zhao_cui_spatial_sir_austria_j9_T20": (
        20,
        "fixed_sir_infectious_components_gaussian_observation_density",
        "sir_log_scale_theta",
        ("log_kappa_scale", "log_nu_scale", "log_obs_noise_scale"),
        (0.0, 0.0, 0.0),
    ),
    "zhao_cui_predator_prey_T20": (
        20,
        "additive_gaussian_predator_prey",
        "physical",
        ("r", "K", "a", "s", "u", "v"),
        (0.6, 114.0, 25.0, 0.3, 0.5, 0.5),
    ),
    "zhao_cui_generalized_sv_synthetic_from_estimated_values": (
        1008,
        "source_route_prior_mean_generalized_sv",
        "source_route_active_transformed_prior_mean",
        ("gamma_unconstrained", "log_tau", "mu"),
        (1.0824113944610982, -2.076793740349318, 0.0),
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def audit(payload: Mapping[str, Any], *, verify_repository_bytes: bool = True) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("independent audit: schema mismatch")
    if tuple(payload.get("main_rows") or ()) != MAIN_ROWS:
        raise ValueError("independent audit: main-row order mismatch")
    if tuple(payload.get("algorithms") or ()) != ALGORITHMS:
        raise ValueError("independent audit: algorithm order mismatch")
    expected_supersession = {
        "authority_amendment_path": AUTHORITY_AMENDMENT_PATH,
        "authority_amendment_sha256": AUTHORITY_AMENDMENT_SHA256,
        "original_phase0_freeze_sha256": ORIGINAL_FREEZE_SHA256,
        "supersession_scope": "sir_target_generation_identity_and_exact_row_extension_classifications_only",
    }
    if payload.get("authority_supersession") != expected_supersession:
        raise ValueError("independent audit: authority-supersession binding mismatch")

    sidecar = _mapping("sidecar", payload.get("sidecar"))
    if (
        sidecar.get("row_id") != SIDECAR_ROW
        or sidecar.get("row_scope") != "scoped_component_row"
        or sidecar.get("target_scope")
        != "local_complete_data_zhao_cui_sir_d18_component"
        or sidecar.get("time_steps") != 20
        or tuple(sidecar.get("parameter_order") or ())
        != ("log_kappa_scale", "log_nu_scale", "log_obs_noise_scale")
        or sidecar.get("closure_required_by_current_program") is not False
        or sidecar.get("target_signature_status")
        != "outside_main_24_cell_program_not_frozen"
        or sidecar.get("counted_as_main_row") is not False
    ):
        raise ValueError("independent audit: sidecar boundary mismatch")

    bindings = payload.get("input_bindings")
    if not isinstance(bindings, list):
        raise ValueError("independent audit: input bindings must be a list")
    observed_bindings = {
        entry.get("path"): (entry.get("sha256"), entry.get("role"))
        for entry in bindings
        if isinstance(entry, Mapping)
    }
    if observed_bindings != EXPECTED_INPUTS:
        raise ValueError("independent audit: input binding manifest mismatch")
    if verify_repository_bytes:
        for relative, (expected_hash, _role) in EXPECTED_INPUTS.items():
            path = ROOT / relative
            if not path.is_file() or _sha256(path) != expected_hash:
                raise ValueError(f"independent audit: repository byte drift: {relative}")

    forward = payload.get("ledh_forward_rows")
    if not isinstance(forward, list) or len(forward) != 6:
        raise ValueError("independent audit: expected six LEDH forward rows")
    by_row = {
        entry.get("row_id"): entry for entry in forward if isinstance(entry, Mapping)
    }
    if tuple(entry.get("row_id") for entry in forward) != MAIN_ROWS:
        raise ValueError("independent audit: LEDH forward-row order mismatch")
    for row, (time_steps, policy, coordinates, names, theta) in ROW_SPECS.items():
        entry = _mapping(f"LEDH row {row}", by_row.get(row))
        observed = (
            entry.get("time_steps"),
            entry.get("num_particles"),
            tuple(entry.get("batch_seeds") or ()),
            entry.get("target_observation_policy"),
            entry.get("theta_coordinate_system"),
            tuple(entry.get("parameter_order") or ()),
            tuple(entry.get("evaluation_theta") or ()),
            entry.get("row_scope"),
            entry.get("full_leaderboard_row"),
        )
        expected = (
            time_steps,
            10000,
            SEEDS,
            policy,
            coordinates,
            names,
            theta,
            "main_observed_data_filtering_row",
            True,
        )
        if observed != expected:
            raise ValueError(f"independent audit: row signature mismatch: {row}")

    cells = payload.get("starting_cells")
    if not isinstance(cells, list) or len(cells) != 24:
        raise ValueError("independent audit: expected 24 starting cells")
    expected_keys = [(row, algorithm) for row in MAIN_ROWS for algorithm in ALGORITHMS]
    if [(cell.get("row_id"), cell.get("algorithm_id")) for cell in cells] != expected_keys:
        raise ValueError("independent audit: starting-cell matrix mismatch")
    for cell in cells:
        row = str(cell.get("row_id"))
        algorithm = str(cell.get("algorithm_id"))
        source = LEDH_HISTORY_SOURCE if algorithm == "ledh_pfpf_ot" else NONLEDH_SOURCE
        closure = (
            "gap_current_source_five_seed_ledh_admission"
            if algorithm == "ledh_pfpf_ot"
            else (
                "frozen_nonledh_baseline_candidate"
                if row in MAIN_ROWS[:3]
                else "gap_target_matched_value_and_score_evaluator"
            )
        )
        if (
            cell.get("status_source_artifact") != source
            or cell.get("status_source_artifact_sha256") != EXPECTED_INPUTS[source][0]
            or cell.get("phase0_closure_status") != closure
            or cell.get("current_program_admitted") is not False
        ):
            raise ValueError(f"independent audit: starting-cell status mismatch: {row}/{algorithm}")

    sidecar_cells = sidecar.get("cells")
    if not isinstance(sidecar_cells, list) or len(sidecar_cells) != 4:
        raise ValueError("independent audit: expected four sidecar cells")
    for index, cell in enumerate(sidecar_cells):
        algorithm = ALGORITHMS[index]
        source = LEDH_HISTORY_SOURCE if algorithm == "ledh_pfpf_ot" else NONLEDH_SOURCE
        expected_closure = {
            "fixed_sgqf": "not_applicable_scoped_sidecar",
            "ukf": "not_applicable_scoped_sidecar",
            "zhao_cui_scalar_or_multistate": (
                "historical_scoped_component_candidate_outside_program"
            ),
            "ledh_pfpf_ot": "historical_scoped_diagnostic_outside_program",
        }[algorithm]
        if (
            cell.get("row_id") != SIDECAR_ROW
            or cell.get("algorithm_id") != algorithm
            or cell.get("status_source_artifact") != source
            or cell.get("status_source_artifact_sha256") != EXPECTED_INPUTS[source][0]
            or cell.get("phase0_closure_status") != expected_closure
            or cell.get("current_program_admitted") is not False
        ):
            raise ValueError("independent audit: sidecar cell mismatch")

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
    if payload.get("summary") != expected_summary:
        raise ValueError("independent audit: summary mismatch")

    policies = _mapping("policies", payload.get("policies"))
    expected_policies = {
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
        "ledh_seed_set": list(SEEDS),
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
    }
    if policies != expected_policies:
        raise ValueError("independent audit: policy freeze mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--skip-repository-byte-check", action="store_true")
    args = parser.parse_args(argv)
    path = args.input if args.input.is_absolute() else ROOT / args.input
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("independent audit: input must be a JSON object")
    audit(payload, verify_repository_bytes=not args.skip_repository_byte_check)
    print(f"PHASE0_INDEPENDENT_AUDIT_PASS {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
