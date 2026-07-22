#!/usr/bin/env python3
"""Build the frozen Phase 0 target and comparator registry for Contract E-TP."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim.ledh_forward_contract import (
    ACTUAL_SV_ROW_ID,
    FIXED_SIR_AUSTRIA_ROW_ID,
    GENERALIZED_SV_PARAMETER_ORDER,
    GENERALIZED_SV_ROW_ID,
    KSC_SV_ROW_ID,
    LGSSM_M3_T50_PARAMETER_ORDER,
    LGSSM_M3_T50_ROW_ID,
    PREDATOR_PREY_PARAMETER_ORDER,
    PREDATOR_PREY_ROW_ID,
    SIR_LOG_SCALE_PARAMETER_ORDER,
    SV_SYNTHETIC_PARAMETER_ORDER,
)
from bayesfilter.highdim.sv_mixture_cut4 import (
    exact_transformed_sv_observations,
    transformed_sv_observations,
)
from docs.benchmarks.benchmark_ledh_same_target_generalized_sv_value import (
    _log_square_flow_observations,
)
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _generalized_sv_prior_mean_dataset,
    _lgssm_dataset,
    _predator_prey_dataset,
    _sir_dataset,
    _sv_dataset,
)


OUTPUT = ROOT / "docs/benchmarks/configs/contract_e_tp_all_models_2026_07_15.json"
MASTER_PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-contract-e-tp-all-model-gradient-comparison-master-plan-2026-07-15.md"
)
ZHAO_CUI_PAPER = ROOT / (
    ".localresources/papers/"
    "zhao-cui-tensor-train-sequential-learning-jmlr-2024.pdf"
)
ZHAO_CUI_TEXT = ROOT / (
    ".localresources/papers/"
    "zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_record(value: Any) -> dict[str, Any]:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    serialized = tf.io.serialize_tensor(tensor).numpy()
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": tensor.dtype.name,
        "serialized_tensor_sha256": hashlib.sha256(serialized).hexdigest(),
        "all_finite": bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()),
    }


def _source_record(path: str) -> dict[str, Any]:
    resolved = ROOT / path
    if not resolved.is_file():
        raise FileNotFoundError(path)
    return {"path": path, "sha256": _sha256_file(resolved)}


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _dataset(
    generator: Callable[[int], dict[str, Any]],
    seed: int,
    transform: Callable[[tf.Tensor], tf.Tensor] | None = None,
) -> tuple[dict[str, Any], list[float]]:
    payload = generator(seed)
    raw = tf.convert_to_tensor(payload["observations"], dtype=tf.float64)
    target = raw if transform is None else transform(raw)
    return {
        "seed": seed,
        "raw_observations": _tensor_record(raw),
        "target_observations": _tensor_record(target),
        "generator": f"{generator.__module__}:{generator.__name__}",
        "transform": (
            "identity"
            if transform is None
            else f"{transform.__module__}:{transform.__name__}"
        ),
    }, [float(value) for value in payload["truth_theta"]]


def _row(
    *,
    row_id: str,
    state_dim: int,
    observation_dim: int,
    horizon: int,
    parameter_order: tuple[str, ...],
    theta_coordinate_system: str,
    truth_theta: list[float],
    dataset: dict[str, Any],
    target_observation_policy: str,
    reference: str,
    zhao_cui_classification: str,
    zhao_cui_status: str,
    zhao_cui_route: str,
    parameter_region: dict[str, Any],
    source_anchors: list[str],
    comparison_scope: str = "primary_observed_data_filtering",
) -> dict[str, Any]:
    if len(parameter_order) != len(truth_theta):
        raise ValueError(f"{row_id}: parameter order and theta disagree")
    return {
        "row_id": row_id,
        "comparison_scope": comparison_scope,
        "state_dim": state_dim,
        "observation_dim": observation_dim,
        "horizon": horizon,
        "target_scalar": "finite_fixed_program_observed_data_log_likelihood",
        "score_target": "total_derivative_of_target_scalar",
        "parameter_order": list(parameter_order),
        "parameter_dim": len(parameter_order),
        "theta_coordinate_system": theta_coordinate_system,
        "truth_theta": truth_theta,
        "dataset": dataset,
        "target_observation_policy": target_observation_policy,
        "initial_law_policy": "bind_existing_row_generator_and_manifest",
        "reference": reference,
        "parameter_region": parameter_region,
        "equivalence_margin_status": "descriptive_only_margin_unavailable",
        "zhao_cui": {
            "classification": zhao_cui_classification,
            "status": zhao_cui_status,
            "route": zhao_cui_route,
            "source_anchors": source_anchors,
            "same_scalar_fd_required": True,
        },
    }


def build_registry() -> dict[str, Any]:
    lgssm_data, lgssm_theta = _dataset(_lgssm_dataset, 81100)
    actual_data, actual_theta = _dataset(
        _sv_dataset, 81101, exact_transformed_sv_observations
    )
    ksc_data, ksc_theta = _dataset(
        _sv_dataset,
        81101,
        lambda value: transformed_sv_observations(value, offset=1.0e-8),
    )
    ksc_data["transform"] = (
        "bayesfilter.highdim.sv_mixture_cut4:transformed_sv_observations"
        "(offset=1e-8)"
    )
    sir_data, sir_theta = _dataset(_sir_dataset, 81103)
    predator_data, predator_theta = _dataset(_predator_prey_dataset, 81104)
    generalized_data, generalized_theta = _dataset(
        _generalized_sv_prior_mean_dataset, 81105
    )
    generalized_raw = tf.convert_to_tensor(
        _generalized_sv_prior_mean_dataset(81105)["observations"],
        dtype=tf.float64,
    )
    generalized_data["proposal_flow_observations"] = _tensor_record(
        _log_square_flow_observations(generalized_raw)
    )
    generalized_data["proposal_flow_transform"] = (
        "docs.benchmarks.benchmark_ledh_same_target_generalized_sv_value:"
        "_log_square_flow_observations"
    )

    region_open = {
        "status": "region_design_required_before_chart_preparation",
        "center": "truth_theta",
        "reason": "no reviewed Contract E-TP chart region exists for this row",
    }
    rows = [
        _row(
            row_id=LGSSM_M3_T50_ROW_ID,
            state_dim=3,
            observation_dim=3,
            horizon=50,
            parameter_order=LGSSM_M3_T50_PARAMETER_ORDER,
            theta_coordinate_system="physical_benchmark_exact_oracle",
            truth_theta=lgssm_theta,
            dataset=lgssm_data,
            target_observation_policy="linear_gaussian_observation",
            reference="exact_differentiated_kalman_filter",
            zhao_cui_classification="fixed_hmc_adaptation",
            zhao_cui_status="missing_real_fixed_tt_route_oracle_adapter_is_not_zhao_cui",
            zhao_cui_route="not_implemented_for_exact_leaderboard_target",
            parameter_region=region_open,
            source_anchors=[
                "Zhao-Cui 2024 Section 6.1",
                "third_party/audit/zhao_cui_tensor_ssm_p10/source/eg1_kalman/main_script.m",
            ],
        ),
        _row(
            row_id=ACTUAL_SV_ROW_ID,
            state_dim=1,
            observation_dim=1,
            horizon=1000,
            parameter_order=SV_SYNTHETIC_PARAMETER_ORDER,
            theta_coordinate_system="synthetic_unconstrained",
            truth_theta=actual_theta,
            dataset=actual_data,
            target_observation_policy="transformed_actual_sv_log_y_square",
            reference="refined_fixed_sgqf_and_high_accuracy_teacher_not_exact",
            zhao_cui_classification="fixed_hmc_adaptation",
            zhao_cui_status="implemented_scalar_adapter_requires_recertification",
            zhao_cui_route="exact_transformed_sv_independent_panel_zhaocui_tt_score",
            parameter_region=region_open,
            source_anchors=[
                "Zhao-Cui 2024 Example 1 and Section 6.2",
                "third_party/audit/zhao_cui_tensor_ssm_p10/source/eg2_sv/mainscript.m",
            ],
        ),
        _row(
            row_id=KSC_SV_ROW_ID,
            state_dim=1,
            observation_dim=1,
            horizon=1000,
            parameter_order=SV_SYNTHETIC_PARAMETER_ORDER,
            theta_coordinate_system="synthetic_unconstrained",
            truth_theta=ksc_theta,
            dataset=ksc_data,
            target_observation_policy="ksc_log_chi_square_gaussian_mixture_surrogate",
            reference="refined_mixture_sgqf_and_high_accuracy_teacher_not_exact",
            zhao_cui_classification="extension_or_invention",
            zhao_cui_status="implemented_fixed_tt_adapter_requires_recertification",
            zhao_cui_route="independent_panel_sv_mixture_zhaocui_tt_score",
            parameter_region=region_open,
            source_anchors=[
                "Zhao-Cui 2024 Example 1 and Section 6.2 for the parent SV model",
                "KSC mixture is a BayesFilter comparator adaptation",
            ],
        ),
        _row(
            row_id=GENERALIZED_SV_ROW_ID,
            state_dim=1,
            observation_dim=1,
            horizon=1008,
            parameter_order=GENERALIZED_SV_PARAMETER_ORDER,
            theta_coordinate_system="source_route_active_transformed_prior_mean",
            truth_theta=generalized_theta,
            dataset=generalized_data,
            target_observation_policy="source_route_prior_mean_generalized_sv",
            reference="native_generalized_sv_reference_and_high_accuracy_teacher",
            zhao_cui_classification="extension_or_invention",
            zhao_cui_status="implemented_fixed_design_extension_requires_recertification",
            zhao_cui_route="scalar_fixed_design_tt_generalized_sv_extension",
            parameter_region=region_open,
            source_anchors=[
                "Zhao-Cui 2024 Example 1 only as parent-model context",
                "generalized-SV operations are not author-source operations",
            ],
        ),
        _row(
            row_id=PREDATOR_PREY_ROW_ID,
            state_dim=2,
            observation_dim=2,
            horizon=20,
            parameter_order=PREDATOR_PREY_PARAMETER_ORDER,
            theta_coordinate_system="physical",
            truth_theta=predator_theta,
            dataset=predator_data,
            target_observation_policy="additive_gaussian_predator_prey",
            reference="dense_and_sgqf_lower_rungs_plus_high_accuracy_teacher",
            zhao_cui_classification="fixed_hmc_adaptation",
            zhao_cui_status="source_model_present_primary_comparator_route_missing",
            zhao_cui_route="forbidden_retained_grid_route_must_not_be_used",
            parameter_region=region_open,
            source_anchors=[
                "Zhao-Cui 2024 Section 6.4",
                "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/pp/odefun.mlx",
                "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/pp/predator_step.mlx",
            ],
        ),
        _row(
            row_id=FIXED_SIR_AUSTRIA_ROW_ID,
            state_dim=18,
            observation_dim=9,
            horizon=20,
            parameter_order=SIR_LOG_SCALE_PARAMETER_ORDER,
            theta_coordinate_system="sir_log_scale_theta",
            truth_theta=sir_theta,
            dataset=sir_data,
            target_observation_policy="additive_gaussian_spatial_sir",
            reference="structural_invariants_lower_dimensional_closures_and_high_accuracy_teacher",
            zhao_cui_classification="fixed_hmc_adaptation_with_logscale_extension",
            zhao_cui_status="implemented_component_score_full_observed_data_total_score_blocked",
            zhao_cui_route="fixed_ttsirt_source_route",
            parameter_region={
                "status": "reviewed_existing_region",
                "box": [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
                "source": "docs/plans/bayesfilter-parameterized-sir-target-contract-2026-07-02.md",
            },
            source_anchors=[
                "Zhao-Cui 2024 Section 6.3",
                "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/sir_austria/odefun.mlx",
                "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/sir_austria/sir_step.mlx",
                "docs/plans/bayesfilter-highdim-zhao-cui-p90-derivative-carry-manifest-2026-06-28.md",
                "docs/plans/bayesfilter-highdim-zhao-cui-p91-phase9-final-decision-result-2026-06-29.md",
            ],
        ),
    ]

    return {
        "schema_version": "contract_e_tp.phase0_registry.v1",
        "program_id": "contract-e-tp-all-model-gradient-comparison",
        "algorithm_id": "contract_e_tp_experimental_v1",
        "status": "phase0_target_source_statistical_freeze",
        "generated_at_git_commit": _git_commit(),
        "device_policy": "cpu_only_registry_build_CUDA_VISIBLE_DEVICES=-1",
        "master_plan": str(MASTER_PLAN.relative_to(ROOT)),
        "primary_row_ids": [row["row_id"] for row in rows],
        "rows": rows,
        "role_seeds": {
            "status": "frozen_disjoint_convenience_design_not_scientific_default",
            "preparation": list(range(82100, 82116)),
            "validation": list(range(82200, 82216)),
            "audit": list(range(82300, 82316)),
            "use": "preparation may select; validation may nominate or veto; audit is final-only",
        },
        "comparison_protocol": {
            "same_scalar_derivative": (
                "AD_manual_JVP_VJP_vs_common_random_number_FD; existing FD policy only"
            ),
            "cross_method_primary": "componentwise_total_score_difference",
            "uncertainty": (
                "paired Student intervals with Bonferroni simultaneous 95 percent component coverage; "
                "paired bootstrap sensitivity when replicate count permits"
            ),
            "pilot_replicates": 16,
            "maximum_replicates_without_plan_revision": 64,
            "precision_rule": (
                "continue 16->32->64 only while interval width can change classification; "
                "otherwise classify unresolved_precision"
            ),
            "equivalence_rule": (
                "row-specific margin must be derived before labels are revealed; rows without "
                "a justified margin remain descriptive_only_margin_unavailable"
            ),
            "forbidden": [
                "0.05*sqrt(p) as cross-method margin",
                "cosine or sign agreement as equivalence",
                "one-seed ranking",
                "retained-grid predator-prey production comparison",
                "P91 SIR component score as full filtering score",
            ],
        },
        "default_assumption_audit": [
            {
                "choice": "mass_mean_second_moment_next_predictive_features",
                "status": "starting_hypothesis_not_cross_model_default",
                "early_diagnostic": "LGSSM per-time feature/tangent ablation",
            },
            {
                "choice": "fixed_square_chart",
                "status": "primary_hypothesis",
                "early_diagnostic": "rank_positivity_and_region_margin",
            },
            {
                "choice": "nonlinear_parameter_regions",
                "status": "open_except_reviewed_SIR_box",
                "early_diagnostic": "phase5_model_specific_region_design",
            },
            {
                "choice": "16_replicate_pilot",
                "status": "owner_precedent_and_convenience_pilot_not_final_power_claim",
                "early_diagnostic": "interval_width_and_precision_stopping",
            },
        ],
        "source_support": {
            "zhao_cui_paper": {
                **_source_record(str(ZHAO_CUI_PAPER.relative_to(ROOT))),
                "text_path": str(ZHAO_CUI_TEXT.relative_to(ROOT)),
                "text_sha256": _sha256_file(ZHAO_CUI_TEXT),
                "classification": "DIRECT_METHOD",
                "publication_status": "JMLR_2024_local_full_text",
                "retraction_erratum_status": "not_checked_against_live_metadata",
                "inspected_anchors": [
                    "Equations 9-14",
                    "Algorithms 1-2",
                    "Algorithms 3-5 and Equations 23, 30-35",
                    "Theorems 7-8",
                    "Sections 6.1-6.4",
                ],
                "allowed_claim": "paper algorithm and example mathematics",
                "forbidden_claim": "BayesFilter fixed-branch score correctness",
            },
            "author_code": {
                **_source_record("third_party/audit/zhao_cui_tensor_ssm_p10/MANIFEST.yml"),
                "upstream_commit": "80034dccb99eb1d86284a1839b4a12067d13b9da",
                "classification": "IMPLEMENTATION_OR_SOFTWARE",
                "allowed_claim": "source operation and example wiring anchors",
                "forbidden_claim": "local derivative correctness or mathematical oracle",
            },
            "contract_e_tp_derivation": {
                **_source_record("docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex"),
                "classification": "PROJECT_DERIVATION",
                "inspected_section": "A proposed score-aware teacher-projection extension",
            },
        },
        "citation_venue_metadata": {
            "zhao_cui_2024": {
                "venue": "Journal of Machine Learning Research",
                "citation_count": "not_available_no_live_metadata_query",
                "venue_metric": "not_collected_not_truth_evidence",
                "access_date": "2026-07-15",
            }
        },
        "backward_snowball": [
            {
                "source": "Zhao-Cui 2024",
                "candidate": "Cui and Dolgov 2022 squared TT and KR rearrangements",
                "classification": "FOUNDATIONAL",
                "action": "inspect before changing fixed-TTSIRT transport mathematics",
            },
            {
                "source": "Zhao-Cui 2024",
                "candidate": "Griebel and Harbrecht 2023 TT approximation bounds",
                "classification": "BACKGROUND",
                "action": "not used as a practical rank guarantee",
            },
        ],
        "forward_snowball": {
            "status": "not_run_no_live_metadata_needed_for_phase0_implementation",
            "blocker": "no approved or necessary live citation-index query",
        },
        "claim_support": [
            {
                "claim": "Zhao-Cui recursively approximates joint posterior densities with TT/squared-TT and marginalizes them",
                "support": "PRIMARY_TECHNICAL_SUPPORT",
                "anchors": "Equations 9-14, Algorithms 1-2, Proposition 2",
            },
            {
                "claim": "Contract E-TP preserves selected finite-teacher feature tangents under a fixed chart",
                "support": "PROJECT_DERIVATION",
                "anchors": "ch32c2 selected-feature tangent proposition",
            },
            {
                "claim": "SIR d18 fixed variant exists but full filtering total score is incomplete",
                "support": "IMPLEMENTATION_EVIDENCE",
                "anchors": "P90 derivative manifest and P91 final decision",
            },
        ],
        "omitted_paper_risks": [
            {
                "risk": "later fixed-TTSIRT derivative implementation may require Cui-Dolgov primary equations",
                "status": "phase6_source_check_required",
            },
            {
                "risk": "no forward-citation/retraction metadata check for Zhao-Cui",
                "status": "does_not_block_local_implementation_plan_but_blocks_complete_literature_claim",
            },
        ],
        "nonclaims": [
            "not implementation correctness",
            "not a cross-method equivalence result",
            "not canonical Contract E status",
            "not leaderboard or HMC readiness",
        ],
    }


def main() -> None:
    registry = build_registry()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
