"""Validate the P0 target-route registry for the DPF filter-oracle program.

This runner is intentionally pure Python.  P0 is a governance/registry phase,
so importing TensorFlow would add environment risk without answering the phase
question.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

REPO_ROOT = Path(__file__).resolve().parents[4]
REGISTRY_PATH = (
    REPO_ROOT
    / "docs"
    / "plans"
    / "bayesfilter-dpf-filter-oracle-comparison-target-route-registry-2026-06-08.json"
)
SUMMARY_PATH = (
    REPO_ROOT
    / "experiments"
    / "dpf_implementation"
    / "reports"
    / "outputs"
    / "dpf_filter_oracle_comparison_p0_target_route_registry_2026-06-08.json"
)
MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_filter_oracle_comparison_p0_registry_tf"
)

REQUIRED_ROUTE_IDS = {
    "kalman_exact",
    "dense_refined_quadrature",
    "ukf",
    "svd_sigma_point",
    "cut4",
    "zhao_cui_fixed_design_tt",
    "dpf_bootstrap_ot",
    "dpf_ledh_pfpf_ot",
}
REQUIRED_CLAIM_CLASSES = {
    "EXACT_ORACLE",
    "CERTIFIED_APPROXIMATION",
    "SURROGATE_USEFULNESS",
    "DIAGNOSTIC_ONLY",
    "BLOCKED",
}
REQUIRED_PHASE_KEYS = {"p1", "p2", "p3", "p4", "p5"}
GRADIENT_STATISTICS = {
    "reference_score",
    "fixed_branch_score",
    "crn_pathwise_score",
    "diagnostic_fixed_branch_score",
    "N/A",
}
REQUIRED_TARGET_FIELDS = {
    "target_id",
    "model_family",
    "target_identity",
    "state_law",
    "observation_law",
    "parameterization",
    "parameter_vector",
    "transformation_jacobian_terms",
    "dimension_panel_convention",
    "governing_sources",
    "target_nonclaims",
}
REQUIRED_ROUTE_FIELDS = {
    "route_status",
    "route_path",
    "value_support",
    "gradient_support",
    "claim_class",
    "promotion_tolerance",
    "certification_band",
    "primary_gradient_statistic",
    "blockers",
    "nonclaims",
    "phase_eligibility",
    "phase_eligibility_reason",
    "seed_evaluator_variance_policy",
}
REQUIRED_MANIFEST_FIELDS = {
    "git_branch",
    "git_commit",
    "scoped_dirty_state_summary",
    "python_version",
    "cpu_gpu_status",
    "cuda_visible_devices",
    "command",
    "wall_time_seconds",
    "timestamp_utc",
    "environment",
    "seeds",
    "particle_counts",
    "data_version",
    "registry_path",
    "summary_path",
    "output_artifact_path",
    "plan_path",
    "result_path",
}


class RegistryValidationError(ValueError):
    """Raised when the registry violates the P0 contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--write-seed-registry", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    if args.write_seed_registry:
        payload = build_seed_registry()
        validate_registry(payload, registry_path=args.registry, require_manifest=False)
    else:
        payload = _load_json(args.registry)
        validate_registry(payload, registry_path=args.registry, require_manifest=True)

    manifest = _run_manifest(
        command=" ".join([sys.executable, "-m", MODULE_PATH, *sys.argv[1:]]),
        registry_path=args.registry,
        summary_path=args.summary,
        wall_time_seconds=time.perf_counter() - start,
    )
    payload["run_manifest"] = manifest
    summary = validate_registry(payload, registry_path=args.registry, require_manifest=True)
    summary["run_manifest"] = manifest

    if args.write_seed_registry or not args.validate_only:
        args.registry.parent.mkdir(parents=True, exist_ok=True)
        args.registry.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if not args.validate_only:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        "P0_REGISTRY_VALID "
        f"targets={summary['target_count']} route_rows={summary['route_row_count']} "
        f"blocked_rows={summary['claim_class_counts'].get('BLOCKED', 0)}"
    )
    return 0


def build_seed_registry() -> dict[str, Any]:
    """Build the reviewed P0 seed registry and expand every target-route pair."""

    target_profiles = _target_profiles()
    targets = []
    route_matrix = {}
    for target, profile in target_profiles:
        targets.append(target)
        route_matrix[target["target_id"]] = _apply_route_policies(
            _routes_for_profile(profile)
        )

    return {
        "schema_version": "dpf_filter_oracle_comparison.target_route_registry.v1",
        "metadata_date": "2026-06-08",
        "phase": "P0",
        "program_id": "dpf-filter-oracle-comparison",
        "purpose": (
            "Governed target-route registry for comparing DPF value and gradient "
            "evidence against exact, approximation, diagnostic, and blocked filter routes."
        ),
        "governing_sources": [
            "docs/plans/bayesfilter-dpf-filter-oracle-comparison-master-program-2026-06-08.md",
            "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p0-target-route-registry-subplan-2026-06-08.md",
            "docs/plans/bayesfilter-highdim-zhao-cui-p42-gradient-likelihood-validation-rules-2026-06-07.md",
            "docs/plans/bayesfilter-highdim-zhao-cui-p45-target-registry-2026-06-08.json",
        ],
        "route_ids": sorted(REQUIRED_ROUTE_IDS),
        "claim_classes": sorted(REQUIRED_CLAIM_CLASSES),
        "gradient_statistics": sorted(GRADIENT_STATISTICS),
        "promotion_tolerance_catalog": {
            "exact_lgssm_tight": {
                "value_abs": 1e-10,
                "relative_score_error": 1e-4,
                "scope": "LGSSM fixture and declared parameterization only.",
            },
            "dense_refined_tiny": {
                "value_refinement_abs": 1e-8,
                "directional_score_refinement_abs": 1e-7,
                "scope": "Tiny scalar/factorized nonlinear fixtures after refinement.",
            },
            "exact_transformed_sv_tight": {
                "value_abs": 2e-8,
                "relative_score_error": 1e-4,
                "scope": "Exact transformed SV dense/reference rows only.",
            },
            "ksc_mixture_tight": {
                "value_abs": 2e-6,
                "relative_score_error": 1e-4,
                "scope": "Finite KSC transformed-mixture target only; not native SV.",
            },
            "p44_local_approximation_band": {
                "value_abs": "target-specific P44 local bound",
                "directional_score_abs": "target-specific P44 local bound",
                "scope": "Local deterministic fixture evidence only.",
            },
            "dpf_p5_pending_mc_band": {
                "mean_value_error_ci": "P5 must predeclare 95% CI and RMSE tolerances.",
                "score_rmse": "P5 must predeclare reference-score comparison tolerance.",
                "scope": "Pending P5 multi-seed particle ladder.",
            },
            "diagnostic_na": {
                "value": "N/A: diagnostic-only row.",
                "gradient": "N/A: diagnostic-only row.",
            },
            "blocked_na": {
                "value": "N/A: blocked row.",
                "gradient": "N/A: blocked row.",
            },
        },
        "certification_band_catalog": {
            "exact_numerical_linear_algebra": (
                "Exact up to reviewed numerical linear algebra and dtype limits."
            ),
            "dense_refinement_band": "Reference refinement band must pass before use as oracle.",
            "exact_transformed_sv_dense_band": "Dense transformed-SV reference and Jacobian tie-out band.",
            "ksc_kalman_mixture_enumeration_band": (
                "Component-enumerated finite-mixture Kalman band for the KSC target."
            ),
            "p44_local_approximation_band": "Existing P44 local deterministic value/score band.",
            "p5_pending_mc_band": "Pending P5 Monte Carlo uncertainty band.",
            "diagnostic_na": "N/A: diagnostic-only row; no certification.",
            "blocked_na": "N/A: blocked row; no certification.",
        },
        "targets": targets,
        "route_matrix": route_matrix,
        "registry_nonclaims": [
            "P0 does not run numerical filters.",
            "P0 does not establish DPF value or gradient closeness.",
            "P0 does not make HMC, production, GPU, paper-scale, or deployment claims.",
            "Rows with implementation-contract evidence are still not oracle evidence unless a same-target reference is named.",
        ],
    }


def validate_registry(
    payload: dict[str, Any],
    *,
    registry_path: Path,
    require_manifest: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    if payload.get("schema_version") != "dpf_filter_oracle_comparison.target_route_registry.v1":
        errors.append("schema_version must be dpf_filter_oracle_comparison.target_route_registry.v1")
    if set(payload.get("route_ids", [])) != REQUIRED_ROUTE_IDS:
        errors.append("route_ids must exactly match the P0 required route IDs")
    if set(payload.get("claim_classes", [])) != REQUIRED_CLAIM_CLASSES:
        errors.append("claim_classes must exactly match the reviewed claim-class set")

    tolerances = _dict_field(payload, "promotion_tolerance_catalog", errors)
    bands = _dict_field(payload, "certification_band_catalog", errors)
    targets = _list_field(payload, "targets", errors)
    route_matrix = _dict_field(payload, "route_matrix", errors)
    if require_manifest:
        _validate_run_manifest(payload.get("run_manifest"), errors)

    target_ids: list[str] = []
    for target in targets:
        if not isinstance(target, dict):
            errors.append("each target must be an object")
            continue
        missing = REQUIRED_TARGET_FIELDS.difference(target)
        if missing:
            errors.append(f"target missing fields {sorted(missing)}: {target.get('target_id')}")
        target_id = str(target.get("target_id", ""))
        if not target_id:
            errors.append("target_id must be nonempty")
        else:
            target_ids.append(target_id)
        for field in REQUIRED_TARGET_FIELDS.difference({"target_id"}):
            if _is_empty(target.get(field)):
                errors.append(f"target {target_id} has empty {field}")

    if len(target_ids) != len(set(target_ids)):
        errors.append("target_id values must be unique")
    if set(route_matrix) != set(target_ids):
        errors.append("route_matrix keys must exactly match target IDs")

    flattened_rows: list[dict[str, Any]] = []
    for target_id in target_ids:
        routes = route_matrix.get(target_id)
        if not isinstance(routes, dict):
            errors.append(f"route_matrix[{target_id}] must be an object")
            continue
        if set(routes) != REQUIRED_ROUTE_IDS:
            errors.append(f"route_matrix[{target_id}] must contain every required route ID")
        for route_id, row in routes.items():
            if route_id not in REQUIRED_ROUTE_IDS:
                errors.append(f"unknown route_id {route_id} for target {target_id}")
                continue
            if not isinstance(row, dict):
                errors.append(f"route row {target_id}/{route_id} must be an object")
                continue
            row_copy = {"target_id": target_id, "route_id": route_id, **row}
            flattened_rows.append(row_copy)
            _validate_route_row(
                target_id=target_id,
                route_id=route_id,
                row=row,
                tolerances=tolerances,
                bands=bands,
                errors=errors,
            )

    expected_rows = len(target_ids) * len(REQUIRED_ROUTE_IDS)
    if len(flattened_rows) != expected_rows:
        errors.append(f"expected {expected_rows} route rows, found {len(flattened_rows)}")

    if errors:
        raise RegistryValidationError("\n".join(f"- {error}" for error in errors))

    claim_counts: dict[str, int] = {claim: 0 for claim in sorted(REQUIRED_CLAIM_CLASSES)}
    phase_counts: dict[str, int] = {phase: 0 for phase in sorted(REQUIRED_PHASE_KEYS)}
    for row in flattened_rows:
        claim_counts[row["claim_class"]] += 1
        for phase, eligible in row["phase_eligibility"].items():
            if eligible:
                phase_counts[phase] += 1

    return {
        "artifact_id": "dpf_filter_oracle_comparison_p0_target_route_registry_2026-06-08",
        "decision": "PASS_P0_REGISTRY_VALIDATION",
        "registry_path": str(registry_path.relative_to(REPO_ROOT)),
        "target_count": len(target_ids),
        "route_id_count": len(REQUIRED_ROUTE_IDS),
        "route_row_count": len(flattened_rows),
        "claim_class_counts": claim_counts,
        "phase_eligibility_counts": phase_counts,
        "dpf_rows_with_seed_policy": sum(
            1
            for row in flattened_rows
            if row["route_id"].startswith("dpf_")
            and not str(row["seed_evaluator_variance_policy"]).startswith("N/A")
        ),
        "nonclaims": [
            "P0 validation is schema and governance validation only.",
            "P0 validation does not run numerical filters.",
            "P0 validation does not establish DPF value or gradient correctness.",
        ],
    }


def _target_profiles() -> list[tuple[dict[str, Any], str]]:
    common_dpf_sources = [
        "experiments/dpf_implementation/tf_tfp/runners/run_v2_bootstrap_ot_values_tf.py",
        "experiments/dpf_implementation/tf_tfp/runners/run_v2_ledh_pfpf_ot_values_tf.py",
        "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-closeout-result-2026-06-07.md",
    ]
    return [
        (
            _target(
                "lgssm_2d_h25_rich",
                "linear_gaussian_state_space",
                "2D LGSSM with horizon 25 from the common V2 fixture.",
                "x_t = A x_{t-1} + q_t, q_t ~ N(0,Q), x_0 ~ N(m0,P0).",
                "y_t = C x_t + r_t, r_t ~ N(0,R).",
                "theta = (transition_matrix_scale, observation_noise_scale) for DPF V2; exact Kalman may also use frozen A,C,Q,R,m0,P0.",
                "(transition_matrix_scale, observation_noise_scale) plus frozen fixture matrices.",
                "None.",
                "state_dim=2, observation_dim=1, horizon=25.",
                [
                    "experiments/dpf_implementation/tf_tfp/fixtures/common_model_suite_tf.py",
                    "experiments/dpf_implementation/tf_tfp/references/kalman_lgssm_tf.py",
                    "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p1-lgssm-exact-oracle-subplan-2026-06-08.md",
                    *common_dpf_sources,
                ],
                [
                    "No nonlinear-model claim.",
                    "No DPF closeness claim before P1/P5 multi-seed evidence.",
                ],
            ),
            "lgssm",
        ),
        (
            _target(
                "p44_m2_cubic_additive_gaussian_panel",
                "tiny_nonlinear_additive_gaussian",
                "Factorized scalar panel with linear Gaussian transition and cubic additive-Gaussian observation.",
                "Per-axis Gaussian AR transition with parameterized rho, q, and initial distribution.",
                "y_t | x_t ~ N(x_t + c x_t^3, R).",
                "theta = (rho_raw, log_q, log_r, raw_initial_mean, cubic_raw).",
                "(rho_raw, log_q, log_r, raw_initial_mean, cubic_raw).",
                "None.",
                "dims 1,2,3 are independent panel sums in the P44 fixture.",
                [
                    "tests/highdim/test_p44_cubic_additive_gaussian.py",
                    "docs/plans/bayesfilter-highdim-zhao-cui-p44-phase2-cubic-additive-gaussian-result-2026-06-07.md",
                ],
                ["No paper-scale nonlinear filtering claim."],
            ),
            "tiny_cubic",
        ),
        (
            _target(
                "p44_m3_quadratic_observation_panel",
                "tiny_nonlinear_additive_gaussian",
                "Factorized scalar panel with linear Gaussian transition and quadratic observation.",
                "Per-axis Gaussian AR transition with parameterized rho, q, and initial distribution.",
                "y_t | x_t ~ N(x_t^2, R), producing symmetric-mode stress.",
                "theta = (rho_raw, log_q, log_r, raw_initial_mean).",
                "(rho_raw, log_q, log_r, raw_initial_mean).",
                "None.",
                "dims 1,2,3 are independent panel sums in the P44 fixture.",
                [
                    "tests/highdim/test_p44_quadratic_observation.py",
                    "docs/plans/bayesfilter-highdim-zhao-cui-p44-phase3-quadratic-observation-result-2026-06-07.md",
                ],
                ["CUT4 is a stress diagnostic on this target, not an oracle."],
            ),
            "tiny_quadratic",
        ),
        (
            _target(
                "p44_m4_nonlinear_transition_h2_panel",
                "tiny_nonlinear_additive_gaussian",
                "Factorized scalar panel with nonlinear transition at horizon 2.",
                "Per-axis nonlinear Gaussian transition from the P44-M4 fixture.",
                "Gaussian observation closure on the transformed state path.",
                "theta = (rho_raw, log_q, log_r, raw_initial_mean, nonlinear_transition_raw).",
                "(rho_raw, log_q, log_r, raw_initial_mean, nonlinear_transition_raw).",
                "None.",
                "dims 1,2,3 are independent panel sums; Zhao-Cui row is horizon-2 only.",
                [
                    "tests/highdim/test_p44_nonlinear_transition.py",
                    "docs/plans/bayesfilter-highdim-zhao-cui-p44-phase4-nonlinear-transition-result-2026-06-07.md",
                ],
                ["No horizon-4 Zhao-Cui promotion for the scalar helper."],
            ),
            "tiny_transition_h2",
        ),
        (
            _target(
                "sv_exact_transformed_log_chi_square_panel",
                "stochastic_volatility_transformed",
                "Exact transformed SV target z_t = log(y_t^2) with observation-only Jacobian tracked separately.",
                "Scalar SV AR law, factorized across panel coordinates.",
                "Exact log-chi-square transformed observation law.",
                "theta = (gamma, beta, sigma) in P41 test ordering.",
                "(gamma, beta, sigma).",
                "Jacobian log|d log(y^2)/dy| is required for raw-native comparisons; exact transformed rows use z only.",
                "dims 1,2,3 are independent panels, not coupled multivariate TT.",
                [
                    "tests/highdim/test_p41_exact_transformed_sv_zhaocui_ladder.py",
                    "docs/plans/bayesfilter-highdim-zhao-cui-p41-exact-transformed-sv-zhaocui-ladder-result-2026-06-07.md",
                ],
                [
                    "No KSC mixture exactness claim.",
                    "No coupled multivariate Zhao-Cui TT claim.",
                ],
            ),
            "exact_transformed_sv",
        ),
        (
            _target(
                "sv_ksc_transformed_mixture_panel",
                "stochastic_volatility_ksc_mixture",
                "Finite KSC transformed-mixture SV approximation target with component enumeration.",
                "Scalar SV AR law, factorized across panel coordinates.",
                "Finite Gaussian-mixture approximation to transformed SV observations.",
                "theta = (gamma, beta, sigma) in P40/P43 test ordering.",
                "(gamma, beta, sigma).",
                "Mixture approximation target only; not the exact transformed or raw-native SV target.",
                "dims 1,2,3 independent panels; component enumeration scales as 7^dim on tiny fixtures.",
                [
                    "tests/highdim/test_p40_sv_kalman_cut4_zhaocui.py",
                    "tests/highdim/test_p43_sv_value_gradient_cut4_zhaocui.py",
                    "docs/plans/bayesfilter-highdim-zhao-cui-p40-sv-kalman-cut4-zhaocui-test-result-2026-06-07.md",
                    "docs/plans/bayesfilter-highdim-zhao-cui-p43-sv-value-gradient-cut4-zhaocui-result-2026-06-07.md",
                ],
                ["No exact native SV claim.", "No KSC importance-reweighting claim."],
            ),
            "ksc_mixture_sv",
        ),
        (
            _target(
                "generalized_sv_native_raw_observation",
                "generalized_stochastic_volatility",
                "Native generalized SV raw-observation likelihood y_t | s_t,h_t,beta.",
                "Independent prior AR laws for s_t and h_t in the tiny fixture; posterior couples them.",
                "y_t = beta s_t + exp(h_t / 2) epsilon_t.",
                "theta = (log_beta) initially; state-law parameters frozen.",
                "(log_beta).",
                "None for raw-y density; transformed rows are separate targets.",
                "one series has latent state (s_t,h_t); panels must be labeled factorized unless reviewed.",
                ["docs/plans/bayesfilter-highdim-zhao-cui-p45-target-registry-2026-06-08.json"],
                ["No native generalized-SV exact filter claim."],
            ),
            "generalized_sv_native",
        ),
        (
            _target(
                "generalized_sv_transformed_residual_diagnostic",
                "generalized_stochastic_volatility",
                "Transformed-residual diagnostic log((y_t - beta s_t)^2) - h_t.",
                "Same fixed independent AR laws for s_t and h_t as the P45 diagnostic fixture.",
                "Diagnostic transformed residual with numerical square floor; not native raw-y likelihood.",
                "theta = (log_beta) initially.",
                "(log_beta).",
                "Exact transform is blocked because residual transform depends on latent s_t.",
                "factorized diagnostic panels only unless separately reviewed.",
                ["docs/plans/bayesfilter-highdim-zhao-cui-p45-target-registry-2026-06-08.json"],
                ["Transformed residual is not exact native generalized SV."],
            ),
            "generalized_sv_transformed",
        ),
        (
            _target(
                "generalized_sv_gaussian_mixture_or_moment_matched_approximation",
                "generalized_stochastic_volatility",
                "Gaussian-mixture or moment-matched approximation row for generalized SV diagnostics.",
                "Fixed independent prior AR laws for s_t and h_t unless later amended.",
                "Approximation route; residual y_t - beta s_t remains state-dependent.",
                "theta = (log_beta) initially.",
                "(log_beta).",
                "Must state whether it approximates log-chi-square residuals or raw conditional variance.",
                "factorized approximation diagnostics only unless separately reviewed.",
                ["docs/plans/bayesfilter-highdim-zhao-cui-p45-target-registry-2026-06-08.json"],
                ["Gaussian mixture is not exact native generalized SV."],
            ),
            "generalized_sv_approximation",
        ),
        (
            _target(
                "spatial_sir_additive_gaussian_closure",
                "spatial_sir",
                "Clean-room additive-Gaussian spatial-SIR closure with infectious-coordinate observations.",
                "P44/P45 closure transition with Gaussian process noise around the SIR mean map.",
                "Gaussian observation noise on infectious coordinates.",
                "theta = (initial_S_shift, initial_I_shift, log_process_scale, log_observation_scale) for J=1 diagnostic closure.",
                "(initial_S_shift, initial_I_shift, log_process_scale, log_observation_scale).",
                "No observation transform; negative-state policy must stay fixed across methods.",
                "J=1 first; J=2,3 are factorized or replicated unless reviewed.",
                [
                    "tests/highdim/test_p30_cut4_statistical_comparators.py",
                    "docs/plans/bayesfilter-highdim-zhao-cui-p45-target-registry-2026-06-08.json",
                    *common_dpf_sources,
                ],
                ["Closure likelihood is not native SIR filtering correctness."],
            ),
            "sir_closure",
        ),
        (
            _target(
                "spatial_sir_native_or_nongaussian_route",
                "spatial_sir",
                "Native or non-Gaussian spatial-SIR likelihood route if later proposed.",
                "Unspecified native stochastic epidemic transition.",
                "Unspecified native or non-Gaussian observation law.",
                "Blocked pending scientific target definition.",
                "Blocked pending native route definition.",
                "Blocked pending native/non-Gaussian target definition.",
                "Not authorized for comparison until a target-definition amendment passes review.",
                ["docs/plans/bayesfilter-highdim-zhao-cui-p45-target-registry-2026-06-08.json"],
                ["No native SIR likelihood correctness claim."],
            ),
            "native_blocked",
        ),
        (
            _target(
                "predator_prey_additive_gaussian_rk4_closure",
                "predator_prey",
                "Tiny additive-Gaussian RK4 predator-prey closure with state (prey,predator).",
                "P44/P45 RK4 transition mean plus Gaussian process noise.",
                "Gaussian observation noise on prey and predator coordinates.",
                "theta = (r,K,a,s,u,v) for the tiny closure fixture.",
                "(r,K,a,s,u,v).",
                "No observation transform; RK4 and positivity diagnostics must stay fixed.",
                "Two-state target first; replicated panels are factorized unless reviewed.",
                [
                    "tests/highdim/test_p30_cut4_statistical_comparators.py",
                    "docs/plans/bayesfilter-highdim-zhao-cui-p45-target-registry-2026-06-08.json",
                    *common_dpf_sources,
                ],
                ["Closure likelihood is not paper-scale predator-prey validation."],
            ),
            "predator_closure",
        ),
        (
            _target(
                "predator_prey_native_or_nongaussian_route",
                "predator_prey",
                "Native or non-Gaussian predator-prey route if later proposed.",
                "Unspecified native/non-Gaussian predator-prey transition.",
                "Unspecified native/non-Gaussian observation law.",
                "Blocked pending scientific target definition.",
                "Blocked pending native route definition.",
                "Blocked pending native/non-Gaussian target definition.",
                "Not authorized for comparison until a target-definition amendment passes review.",
                ["docs/plans/bayesfilter-highdim-zhao-cui-p45-target-registry-2026-06-08.json"],
                ["No native predator-prey likelihood correctness claim."],
            ),
            "native_blocked",
        ),
    ]


def _target(
    target_id: str,
    model_family: str,
    target_identity: str,
    state_law: str,
    observation_law: str,
    parameterization: str,
    parameter_vector: str,
    transformation_jacobian_terms: str,
    dimension_panel_convention: str,
    governing_sources: list[str],
    target_nonclaims: list[str],
) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "model_family": model_family,
        "target_identity": target_identity,
        "state_law": state_law,
        "observation_law": observation_law,
        "parameterization": parameterization,
        "parameter_vector": parameter_vector,
        "transformation_jacobian_terms": transformation_jacobian_terms,
        "dimension_panel_convention": dimension_panel_convention,
        "governing_sources": governing_sources,
        "target_nonclaims": target_nonclaims,
    }


def _routes_for_profile(profile: str) -> dict[str, dict[str, Any]]:
    if profile == "lgssm":
        return {
            "kalman_exact": _exact(
                "available_exact_kalman_value_gradient_reference_pending_p1_binding",
                "experiments/dpf_implementation/tf_tfp/references/kalman_lgssm_tf.py::run_kalman_filter_tf; analytic-gradient binding in P1",
                "Exact LGSSM log likelihood.",
                "reference_score from Kalman analytic recursion in the declared parameterization.",
                "exact_lgssm_tight",
                "exact_numerical_linear_algebra",
                ("p1", "p5"),
            ),
            "dense_refined_quadrature": _blocked("LGSSM H25 2D dense quadrature is unnecessary and not approved as the exact reference."),
            "ukf": _diag(
                "available_same_target_affine_sigma_point_sanity",
                "bayesfilter/nonlinear/sigma_points_tf.py::tf_svd_sigma_point_filter backend=tf_svd_ukf",
                "Affine LGSSM sanity route; Kalman remains the oracle.",
                "diagnostic fixed-branch score only if P1 binds derivatives.",
                ("p1",),
            ),
            "svd_sigma_point": _diag(
                "available_same_target_affine_cubature_sanity",
                "bayesfilter/nonlinear/sigma_points_tf.py::tf_svd_sigma_point_filter backend=tf_svd_cubature",
                "Affine LGSSM sanity route; Kalman remains the oracle.",
                "diagnostic fixed-branch score only if P1 binds derivatives.",
                ("p1",),
            ),
            "cut4": _diag(
                "available_same_target_affine_cut4_sanity",
                "bayesfilter/nonlinear/svd_cut_tf.py::tf_svd_cut4_filter",
                "Affine LGSSM CUT4 sanity route; Kalman remains the oracle.",
                "diagnostic fixed-branch score only if P1 binds derivatives.",
                ("p1",),
            ),
            "zhao_cui_fixed_design_tt": _diag(
                "available_clean_room_exact_reference_sanity_only",
                "bayesfilter/highdim/filtering.py::FixedBranchSquaredTTFilter",
                "Clean-room LGSSM value-path sanity route; Kalman remains the oracle.",
                "diagnostic fixed-branch score only after explicit P1 binding.",
                ("p1", "p4"),
            ),
            "dpf_bootstrap_ot": _dpf(
                "eligible_for_p1_p5_kalman_oracle_comparison",
                "experiments/dpf_implementation/tf_tfp/runners/run_v2_bootstrap_ot_values_tf.py",
                ("p1", "p5"),
            ),
            "dpf_ledh_pfpf_ot": _dpf(
                "eligible_for_p1_p5_kalman_oracle_comparison",
                "experiments/dpf_implementation/tf_tfp/runners/run_v2_ledh_pfpf_ot_values_tf.py",
                ("p1", "p5"),
            ),
        }
    if profile in {"tiny_cubic", "tiny_quadratic", "tiny_transition_h2"}:
        target_label = profile.replace("tiny_", "p44_")
        cut4_claim = "CERTIFIED_APPROXIMATION"
        cut4_tol = "p44_local_approximation_band"
        cut4_band = "p44_local_approximation_band"
        cut4_status = "available_local_same_target_approximation"
        cut4_nonclaims = ["CUT4 is not an exact nonlinear oracle."]
        if profile == "tiny_quadratic":
            cut4_claim = "DIAGNOSTIC_ONLY"
            cut4_tol = "diagnostic_na"
            cut4_band = "diagnostic_na"
            cut4_status = "available_stress_diagnostic_only"
            cut4_nonclaims = ["Quadratic-observation CUT4 stress gap is diagnostic, not a certificate."]
        return {
            "kalman_exact": _blocked("Nonlinear target; Kalman is exact only for nested linear sanity rows."),
            "dense_refined_quadrature": _exact(
                "available_refined_dense_reference",
                f"tests/highdim/test_p44_{_p44_file_suffix(profile)}.py::_dense_panel_value",
                "Refined dense quadrature reference for the declared tiny target.",
                "reference_score via AD through the refined dense value path.",
                "dense_refined_tiny",
                "dense_refinement_band",
                ("p2", "p5"),
            ),
            "ukf": _diag(
                "available_gaussian_sigma_point_diagnostic",
                "bayesfilter/nonlinear/sigma_points_tf.py::tf_svd_sigma_point_filter backend=tf_svd_ukf",
                "Gaussian sigma-point approximation diagnostic.",
                "diagnostic fixed-branch score only.",
                ("p2",),
            ),
            "svd_sigma_point": _diag(
                "available_cubature_sigma_point_diagnostic",
                "bayesfilter/nonlinear/sigma_points_tf.py::tf_svd_sigma_point_filter backend=tf_svd_cubature",
                "Cubature sigma-point approximation diagnostic.",
                "diagnostic fixed-branch score only.",
                ("p2",),
            ),
            "cut4": _row(
                cut4_status,
                "bayesfilter/nonlinear/svd_cut_tf.py::tf_svd_cut4_filter",
                "Local deterministic same-target approximation evidence from P44.",
                "diagnostic_fixed_branch_score",
                cut4_claim,
                cut4_tol,
                cut4_band,
                "diagnostic fixed-branch score compared to dense where P44 reports it.",
                [],
                cut4_nonclaims,
                ("p2",),
            ),
            "zhao_cui_fixed_design_tt": _cert(
                "available_local_fixed_design_tt_approximation",
                "bayesfilter/highdim/filtering.py::scalar_nonlinear_fixed_design_tt_value_path",
                f"Local fixed-design TT approximation for {target_label}.",
                "fixed-branch AD score compared to dense within the local P44 band.",
                "p44_local_approximation_band",
                "p44_local_approximation_band",
                ("p2", "p4"),
            ),
            "dpf_bootstrap_ot": _dpf(
                "eligible_after_p2_dense_reference_pass",
                "experiments/dpf_implementation/tf_tfp/filters/dpf_ot_tf.py::run_ot_dpf_tf",
                ("p5",),
            ),
            "dpf_ledh_pfpf_ot": _dpf(
                "eligible_after_p2_dense_reference_pass",
                "experiments/dpf_implementation/tf_tfp/filters/ledh_pfpf_ot_tf.py::run_ledh_pfpf_ot_tf",
                ("p5",),
            ),
        }
    if profile == "exact_transformed_sv":
        return {
            "kalman_exact": _blocked("Exact transformed log-chi-square SV is not linear Gaussian; KSC mixture uses a separate target row."),
            "dense_refined_quadrature": _exact(
                "available_exact_transformed_dense_reference",
                "bayesfilter/highdim/sv_mixture_cut4.py::exact_transformed_sv_independent_panel_dense_reference",
                "Exact transformed SV dense reference.",
                "reference_score where P43 gradient ladder binds the transformed target.",
                "exact_transformed_sv_tight",
                "exact_transformed_sv_dense_band",
                ("p3",),
            ),
            "ukf": _blocked("No same-target UKF transformed log-chi-square route is approved."),
            "svd_sigma_point": _blocked("No same-target cubature transformed log-chi-square route is approved."),
            "cut4": _blocked("CUT4 route belongs to the KSC mixture target row, not exact transformed SV."),
            "zhao_cui_fixed_design_tt": _cert(
                "available_exact_transformed_fixed_design_tt_local_certificate",
                "bayesfilter/highdim/sv_mixture_cut4.py::exact_transformed_sv_independent_panel_zhaocui_tt_filter",
                "Fixed-design TT matches exact transformed dense reference on tiny panels.",
                "fixed-branch score only where P43 binds the transformed target.",
                "exact_transformed_sv_tight",
                "exact_transformed_sv_dense_band",
                ("p3", "p4"),
            ),
            "dpf_bootstrap_ot": _blocked("No DPF adapter is approved for the exact transformed SV target in P0."),
            "dpf_ledh_pfpf_ot": _blocked("No LEDH-PFPF adapter is approved for the exact transformed SV target in P0."),
        }
    if profile == "ksc_mixture_sv":
        return {
            "kalman_exact": _exact(
                "available_component_enumerated_kalman_mixture_oracle",
                "bayesfilter/highdim/sv_mixture_cut4.py::independent_panel_sv_mixture_kalman_filter",
                "Exact finite-mixture Kalman evaluation for the KSC approximation target.",
                "reference_score for the KSC finite-mixture target in P43.",
                "ksc_mixture_tight",
                "ksc_kalman_mixture_enumeration_band",
                ("p3",),
            ),
            "dense_refined_quadrature": _diag(
                "available_scalar_dense_context_only",
                "tests/highdim/test_p39_sv_mixture_cut4.py",
                "Dense context for scalar components; not the primary P40 oracle.",
                "diagnostic score only.",
                ("p3",),
            ),
            "ukf": _blocked("No reviewed UKF finite-mixture enumeration route is approved."),
            "svd_sigma_point": _blocked("No reviewed cubature finite-mixture enumeration route is approved."),
            "cut4": _cert(
                "available_same_target_ksc_cut4_certificate",
                "bayesfilter/highdim/sv_mixture_cut4.py::independent_panel_sv_mixture_cut4_filter",
                "CUT4 matches the component-enumerated KSC Kalman mixture target locally.",
                "fixed-branch score in the P43 KSC ladder.",
                "ksc_mixture_tight",
                "ksc_kalman_mixture_enumeration_band",
                ("p3",),
            ),
            "zhao_cui_fixed_design_tt": _blocked("No transformed-mixture Zhao-Cui/TT lane is approved."),
            "dpf_bootstrap_ot": _blocked("No DPF adapter is approved for the KSC mixture approximation target in P0."),
            "dpf_ledh_pfpf_ot": _blocked("No LEDH-PFPF adapter is approved for the KSC mixture approximation target in P0."),
        }
    if profile == "generalized_sv_native":
        return {route: _blocked("Native generalized-SV same-target reference and route contract are missing.") for route in REQUIRED_ROUTE_IDS}
    if profile == "generalized_sv_transformed":
        rows = {route: _blocked("Transformed generalized-SV residual is diagnostic only and lacks same-target oracle.") for route in REQUIRED_ROUTE_IDS}
        rows["cut4"] = _diag(
            "available_transformed_residual_diagnostic_only",
            "docs/plans/bayesfilter-highdim-zhao-cui-p45-target-registry-2026-06-08.json",
            "Finite transformed-residual diagnostic only.",
            "diagnostic score only; no native equality.",
            (),
        )
        return rows
    if profile == "generalized_sv_approximation":
        rows = {route: _blocked("Exact generalized-SV approximation target is not fixed enough for same-target comparison.") for route in REQUIRED_ROUTE_IDS}
        rows["kalman_exact"] = _row(
            "candidate_surrogate_only_pending_target_definition",
            "future KSC or moment-matched Kalman approximation row",
            "Approximate surrogate value only after target definition.",
            "N/A",
            "SURROGATE_USEFULNESS",
            "diagnostic_na",
            "diagnostic_na",
            "N/A: no gradient comparison is authorized.",
            [],
            ["Surrogate does not equal native generalized SV."],
            (),
        )
        rows["cut4"] = _row(
            "candidate_surrogate_only_pending_target_definition",
            "future CUT4 approximation row",
            "Approximate surrogate value only after target definition.",
            "diagnostic_fixed_branch_score",
            "SURROGATE_USEFULNESS",
            "diagnostic_na",
            "diagnostic_na",
            "diagnostic score only.",
            [],
            ["Surrogate does not equal native generalized SV."],
            (),
        )
        return rows
    if profile in {"sir_closure", "predator_closure"}:
        return {
            "kalman_exact": _blocked("Additive-Gaussian nonlinear closure is not linear Gaussian; Kalman is not exact."),
            "dense_refined_quadrature": _blocked("Dense/refined same-target closure reference is pending."),
            "ukf": _diag(
                "available_structural_sigma_point_diagnostic_only",
                "bayesfilter/nonlinear/sigma_points_tf.py::tf_svd_sigma_point_filter backend=tf_svd_ukf",
                "Structural Gaussian sigma-point diagnostic only.",
                "diagnostic fixed-branch score only.",
                (),
            ),
            "svd_sigma_point": _diag(
                "available_structural_cubature_diagnostic_only",
                "bayesfilter/nonlinear/sigma_points_tf.py::tf_svd_sigma_point_filter backend=tf_svd_cubature",
                "Structural cubature diagnostic only.",
                "diagnostic fixed-branch score only.",
                (),
            ),
            "cut4": _diag(
                "available_cut4_closure_diagnostic_only",
                "bayesfilter/nonlinear/svd_cut_tf.py::tf_svd_cut4_filter",
                "Finite closure diagnostic only.",
                "diagnostic fixed-branch score only.",
                (),
            ),
            "zhao_cui_fixed_design_tt": _blocked("P46 multistate adapter does not promote this closure without same-target dense/CUT4 gates."),
            "dpf_bootstrap_ot": _diag(
                "available_v2_dpf_contract_value_context_only",
                "experiments/dpf_implementation/tf_tfp/runners/run_v2_bootstrap_ot_values_tf.py",
                "DPF implementation-contract context only; no closure oracle.",
                "fixed-branch gradient context only where V2 includes the row.",
                (),
            ),
            "dpf_ledh_pfpf_ot": _diag(
                "available_v2_ledh_contract_value_context_only",
                "experiments/dpf_implementation/tf_tfp/runners/run_v2_ledh_pfpf_ot_values_tf.py",
                "LEDH implementation-contract context only; no closure oracle.",
                "fixed-branch gradient context only where V2 includes the row.",
                (),
            ),
        }
    if profile == "native_blocked":
        return {route: _blocked("Native or non-Gaussian route is unspecified and unimplemented.") for route in REQUIRED_ROUTE_IDS}
    raise ValueError(f"unknown profile: {profile}")


def _apply_route_policies(routes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for route_id, row in routes.items():
        phase_eligibility = row.get("phase_eligibility", {})
        dpf_execution_row = route_id.startswith("dpf_") and (
            phase_eligibility.get("p1") or phase_eligibility.get("p5")
        )
        row["seed_evaluator_variance_policy"] = (
            "paired_seeds_min_5; particle_counts_min_2; report evaluator variance, RMSE, max error, and 95% CI"
            if dpf_execution_row
            else "N/A: deterministic, blocked, or not P1/P5 DPF execution row"
        )
    return routes


def _p44_file_suffix(profile: str) -> str:
    if profile == "tiny_cubic":
        return "cubic_additive_gaussian"
    if profile == "tiny_quadratic":
        return "quadratic_observation"
    if profile == "tiny_transition_h2":
        return "nonlinear_transition"
    raise ValueError(profile)


def _exact(
    route_status: str,
    route_path: str,
    value_support: str,
    gradient_support: str,
    promotion_tolerance: str,
    certification_band: str,
    phases: tuple[str, ...],
) -> dict[str, Any]:
    return _row(
        route_status,
        route_path,
        value_support,
        "reference_score",
        "EXACT_ORACLE",
        promotion_tolerance,
        certification_band,
        gradient_support,
        [],
        ["Exactness is scoped to the declared target and fixture only."],
        phases,
    )


def _cert(
    route_status: str,
    route_path: str,
    value_support: str,
    gradient_support: str,
    promotion_tolerance: str,
    certification_band: str,
    phases: tuple[str, ...],
) -> dict[str, Any]:
    return _row(
        route_status,
        route_path,
        value_support,
        "fixed_branch_score",
        "CERTIFIED_APPROXIMATION",
        promotion_tolerance,
        certification_band,
        gradient_support,
        [],
        ["Certificate is local and does not transfer to other targets or dimensions."],
        phases,
    )


def _diag(
    route_status: str,
    route_path: str,
    value_support: str,
    gradient_support: str,
    phases: tuple[str, ...],
) -> dict[str, Any]:
    return _row(
        route_status,
        route_path,
        value_support,
        "diagnostic_fixed_branch_score",
        "DIAGNOSTIC_ONLY",
        "diagnostic_na",
        "diagnostic_na",
        gradient_support,
        [],
        ["Diagnostic row cannot promote correctness or oracle status."],
        phases,
    )


def _dpf(route_status: str, route_path: str, phases: tuple[str, ...]) -> dict[str, Any]:
    return _row(
        route_status,
        route_path,
        "Existing implementation-contract evidence is context only; oracle closeness is pending.",
        "fixed_branch_score",
        "DIAGNOSTIC_ONLY",
        "dpf_p5_pending_mc_band",
        "p5_pending_mc_band",
        "Fixed-branch score only unless a reviewed stochastic-score amendment is accepted.",
        [],
        [
            "No stochastic-resampling distribution correctness claim.",
            "No DPF oracle closeness claim before P1/P5.",
        ],
        phases,
    )


def _blocked(reason: str) -> dict[str, Any]:
    return _row(
        "blocked",
        f"N/A: {reason}",
        "N/A: route is blocked for this target.",
        "N/A",
        "BLOCKED",
        "blocked_na",
        "blocked_na",
        "N/A: route is blocked for this target.",
        [reason],
        ["Blocked row is explicit; absence of execution is not a pass."],
        (),
    )


def _row(
    route_status: str,
    route_path: str,
    value_support: str,
    primary_gradient_statistic: str,
    claim_class: str,
    promotion_tolerance: str,
    certification_band: str,
    gradient_support: str,
    blockers: list[str],
    nonclaims: list[str],
    phases: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "route_status": route_status,
        "route_path": route_path,
        "value_support": value_support,
        "gradient_support": gradient_support,
        "claim_class": claim_class,
        "promotion_tolerance": promotion_tolerance,
        "certification_band": certification_band,
        "primary_gradient_statistic": primary_gradient_statistic,
        "blockers": blockers,
        "nonclaims": nonclaims,
        "phase_eligibility": {phase: phase in phases for phase in sorted(REQUIRED_PHASE_KEYS)},
        "phase_eligibility_reason": (
            f"Eligible for {', '.join(phases)} under P0 routing only; execution still needs phase gate."
            if phases
            else "Not eligible for P1-P5 execution under P0 routing."
        ),
        "seed_evaluator_variance_policy": "N/A: route policy is assigned from route_id after row construction",
    }


def _validate_route_row(
    *,
    target_id: str,
    route_id: str,
    row: dict[str, Any],
    tolerances: dict[str, Any],
    bands: dict[str, Any],
    errors: list[str],
) -> None:
    missing = REQUIRED_ROUTE_FIELDS.difference(row)
    if missing:
        errors.append(f"route row {target_id}/{route_id} missing fields {sorted(missing)}")
        return

    for field in REQUIRED_ROUTE_FIELDS:
        if field == "blockers":
            continue
        if _is_empty(row.get(field)):
            errors.append(f"route row {target_id}/{route_id} has empty {field}")

    claim_class = row.get("claim_class")
    if claim_class not in REQUIRED_CLAIM_CLASSES:
        errors.append(f"route row {target_id}/{route_id} has invalid claim_class {claim_class}")

    if row.get("promotion_tolerance") not in tolerances:
        errors.append(f"route row {target_id}/{route_id} references unknown promotion_tolerance")
    if row.get("certification_band") not in bands:
        errors.append(f"route row {target_id}/{route_id} references unknown certification_band")

    statistic = row.get("primary_gradient_statistic")
    if statistic not in GRADIENT_STATISTICS:
        errors.append(f"route row {target_id}/{route_id} has invalid primary_gradient_statistic")

    phase_eligibility = row.get("phase_eligibility")
    if not isinstance(phase_eligibility, dict) or set(phase_eligibility) != REQUIRED_PHASE_KEYS:
        errors.append(f"route row {target_id}/{route_id} must include P1-P5 phase eligibility")
    elif any(not isinstance(value, bool) for value in phase_eligibility.values()):
        errors.append(f"route row {target_id}/{route_id} phase eligibility values must be booleans")

    blockers = row.get("blockers")
    if not isinstance(blockers, list):
        errors.append(f"route row {target_id}/{route_id} blockers must be a list")
    if claim_class == "BLOCKED" and not blockers:
        errors.append(f"blocked route row {target_id}/{route_id} must list blockers")
    if str(row.get("route_status", "")).startswith("blocked") and claim_class != "BLOCKED":
        errors.append(f"blocked status must use BLOCKED claim class for {target_id}/{route_id}")
    if claim_class in {"EXACT_ORACLE", "CERTIFIED_APPROXIMATION"} and statistic == "N/A":
        errors.append(f"promoted route row {target_id}/{route_id} must name a gradient statistic")
    if route_id.startswith("dpf_"):
        policy = str(row.get("seed_evaluator_variance_policy", ""))
        if (phase_eligibility or {}).get("p1") or (phase_eligibility or {}).get("p5"):
            if policy.startswith("N/A"):
                errors.append(f"DPF execution row {target_id}/{route_id} needs seed/evaluator policy")
            if row.get("primary_gradient_statistic") not in {
                "fixed_branch_score",
                "crn_pathwise_score",
                "diagnostic_fixed_branch_score",
            }:
                errors.append(f"DPF execution row {target_id}/{route_id} needs DPF gradient statistic")
        elif not policy.startswith("N/A"):
            errors.append(f"non-execution DPF row {target_id}/{route_id} must not carry execution seed policy")
    elif not str(row.get("seed_evaluator_variance_policy", "")).startswith("N/A"):
        errors.append(f"non-DPF row {target_id}/{route_id} must not carry DPF seed/evaluator policy")

    nonclaims = row.get("nonclaims")
    if not isinstance(nonclaims, list) or not nonclaims:
        errors.append(f"route row {target_id}/{route_id} must list nonclaims")


def _validate_run_manifest(value: Any, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append("run_manifest must be a top-level object")
        return
    missing = REQUIRED_MANIFEST_FIELDS.difference(value)
    if missing:
        errors.append(f"run_manifest missing fields {sorted(missing)}")
    for field in REQUIRED_MANIFEST_FIELDS:
        if _is_empty(value.get(field)):
            errors.append(f"run_manifest has empty {field}")


def _dict_field(payload: dict[str, Any], field: str, errors: list[str]) -> dict[str, Any]:
    value = payload.get(field)
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object")
        return {}
    return value


def _list_field(payload: dict[str, Any], field: str, errors: list[str]) -> list[Any]:
    value = payload.get(field)
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")
        return []
    return value


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RegistryValidationError("registry root must be an object")
    return loaded


def _run_manifest(
    *,
    command: str,
    registry_path: Path,
    summary_path: Path,
    wall_time_seconds: float,
) -> dict[str, Any]:
    registry_rel = str(registry_path.relative_to(REPO_ROOT))
    summary_rel = str(summary_path.relative_to(REPO_ROOT))
    return {
        "git_branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": _git(["git", "rev-parse", "HEAD"]),
        "scoped_dirty_state_summary": _git(
            [
                "git",
                "status",
                "--short",
                "--",
                registry_rel,
                summary_rel,
                "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p0-target-route-registry-result-2026-06-08.md",
                "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p0-claude-review-ledger-2026-06-08.md",
                "docs/plans/bayesfilter-dpf-filter-oracle-comparison-visible-execution-ledger-2026-06-08.md",
                "experiments/dpf_implementation/tf_tfp/runners/run_filter_oracle_comparison_p0_registry_tf.py",
            ]
        )
        or "clean_for_p0_paths",
        "python_version": platform.python_version(),
        "environment": "active Python environment; observed executable is recorded in command",
        "cpu_gpu_status": "pure_python_cpu_only; CUDA_VISIBLE_DEVICES=-1 set before validation; TensorFlow not imported",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "command": command,
        "wall_time_seconds": wall_time_seconds,
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "seeds": "N/A: P0 is schema/governance validation only",
        "particle_counts": "N/A: P0 does not run DPF filters",
        "data_version": "N/A: no external data loaded",
        "registry_path": registry_rel,
        "summary_path": summary_rel,
        "output_artifact_path": summary_rel,
        "plan_path": "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p0-target-route-registry-subplan-2026-06-08.md",
        "result_path": "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p0-target-route-registry-result-2026-06-08.md",
    }


def _git(args: list[str]) -> str:
    completed = subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
