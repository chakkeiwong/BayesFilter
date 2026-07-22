#!/usr/bin/env python3
"""Build P0 registry artifacts for the multi-model NeuTra program."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROGRAM_ID = "multimodel-neutra-filter-posterior-20260715"
SCHEMA_VERSION = "bayesfilter.multimodel_neutra_p0.v1"
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-multimodel-neutra-filter-posterior-p0-target-route-freeze-"
    "subplan-2026-07-15.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-multimodel-neutra-filter-posterior-p0-target-route-freeze-"
    "result-2026-07-15.md"
)

SOURCE_PATHS = (
    ".localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.pdf",
    ".localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt",
    "third_party/audit/tensor-ssm-paper-demo/models/full_sol.m",
    "third_party/audit/tensor-ssm-paper-demo/models/pre_sol.m",
    "third_party/audit/tensor-ssm-paper-demo/eg2_sv/mainscript.m",
    "third_party/audit/tensor-ssm-paper-demo/eg3_sir/mainscript.m",
    "third_party/audit/tensor-ssm-paper-demo/eg4_predatorprey/mainscript.m",
    "third_party/audit/tensor-ssm-paper-demo/deep-tensor.dev/src/@TTSIRT/marginalise.m",
    "third_party/audit/tensor-ssm-paper-demo/deep-tensor.dev/src/@TTFun/cross.m",
    "bayesfilter/highdim/sv_mixture_cut4.py",
    "bayesfilter/highdim/models.py",
    "bayesfilter/highdim/filtering.py",
    "bayesfilter/highdim/source_route.py",
    "bayesfilter/highdim/zhao_cui_fixed_adjacent_tt_tf.py",
    "bayesfilter/testing/structural_fixtures.py",
    "tests/highdim/test_p47_predator_prey_filtering.py",
    "tests/test_structural_sigma_points.py",
    "docs/chapters/ch18b_structural_deterministic_dynamics.tex",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "UNKNOWN"
    return completed.stdout.strip()


def _cell(
    *,
    cell_id: str,
    model_target: str,
    filter_likelihood: str,
    route_classification: str,
    current_routes: list[str],
    known_parameterization: dict[str, Any],
    known_observations: dict[str, Any],
    blockers: list[dict[str, str]],
    source_anchors: list[str] | None = None,
) -> dict[str, Any]:
    scope_payload = {
        "program_id": PROGRAM_ID,
        "cell_id": cell_id,
        "model_target": model_target,
        "filter_likelihood": filter_likelihood,
        "route_classification": route_classification,
        "current_routes": current_routes,
        "known_parameterization": known_parameterization,
        "known_observations": known_observations,
        "blocker_codes": [item["code"] for item in blockers],
    }
    return {
        "cell_id": cell_id,
        "state": "TARGET_BLOCKED",
        "scope_identity": _canonical_hash(scope_payload),
        "scope_identity_role": (
            "inventory identity only; not a posterior target signature and not "
            "admissible for HMC, training, or transport loading"
        ),
        "target_signature": None,
        "target_signature_status": "NOT_ISSUED_INCOMPLETE_POSTERIOR_CONTRACT",
        "model_target": model_target,
        "filter_likelihood": filter_likelihood,
        "route_classification": route_classification,
        "current_routes": current_routes,
        "known_parameterization": known_parameterization,
        "known_observations": known_observations,
        "source_anchors": source_anchors or [],
        "blockers": blockers,
        "earliest_reentry_rung": "P0_TARGET_FREEZE",
        "forbidden_substitutions": (
            "another filter posterior",
            "another model family",
            "likelihood-only identity",
            "complete-data density",
            "scout or diagnostic route",
        ),
    }


def _cells() -> list[dict[str, Any]]:
    common_posterior_blockers = [
        {
            "code": "MISSING_FROZEN_PARAMETER_PRIOR",
            "detail": "No program-reviewed parameter prior is bound to this filter likelihood.",
        },
        {
            "code": "MISSING_FROZEN_DATA_IDENTITY",
            "detail": "No serious observation tensor/data hash is frozen for this cell.",
        },
        {
            "code": "MISSING_POSTERIOR_RECOMPOSITION",
            "detail": "No independent total posterior value/score recomposition dossier exists.",
        },
    ]
    sv_parameterization = {
        "dimension": 2,
        "unconstrained_order": ["normal_quantile_gamma", "log_beta"],
        "physical_map": ["gamma=Phi(theta[0])", "beta=exp(theta[1])"],
        "sigma": "fixed but serious value not frozen",
        "chart_status": "implemented_for_likelihood; posterior prior/Jacobian convention not frozen",
    }
    sv_observations = {
        "candidate_fixture": "synthetic seed 81101, T=1000 in existing benchmark",
        "exact_transform": "z=log(y^2), requires nonzero y",
        "serious_data_status": "not frozen",
    }
    paper_algorithms = [
        "paper Section 2.1 equations (9)-(11)",
        "paper Algorithm 1 equation (12)",
        "paper Proposition 2 and Algorithm 2",
        "paper Algorithm 5",
    ]
    author_recursion = [
        "third_party/audit/tensor-ssm-paper-demo/models/full_sol.m:21",
        "third_party/audit/tensor-ssm-paper-demo/models/full_sol.m:46",
        "third_party/audit/tensor-ssm-paper-demo/models/pre_sol.m:16",
        "third_party/audit/tensor-ssm-paper-demo/models/pre_sol.m:33",
    ]
    rows = [
        _cell(
            cell_id="SVX-SGQF",
            model_target="exact transformed non-Gaussian stochastic volatility",
            filter_likelihood="fixed SGQF direct-likelihood approximation",
            route_classification="BAYESFILTER_APPROXIMATION",
            current_routes=[
                "bayesfilter/highdim/sv_mixture_cut4.py:663",
                "bayesfilter/highdim/sv_mixture_cut4.py:771",
            ],
            known_parameterization=sv_parameterization,
            known_observations=sv_observations,
            blockers=[
                *common_posterior_blockers,
                {
                    "code": "MISSING_SERIOUS_SGQF_SETTINGS",
                    "detail": "Sparse level/cloud and target-region admission settings are not frozen.",
                },
                {
                    "code": "MISSING_BATCH_NATIVE_POSTERIOR_ADAPTER",
                    "detail": "Current filter functions are not a registered batched posterior adapter.",
                },
            ],
        ),
        _cell(
            cell_id="SVX-ZC",
            model_target="exact transformed non-Gaussian stochastic volatility",
            filter_likelihood="factorized scalar fixed-design TT approximation",
            route_classification="EXTENSION_OR_INVENTION_CURRENT_WRAPPER",
            current_routes=[
                "bayesfilter/highdim/sv_mixture_cut4.py:1267",
                "bayesfilter/highdim/sv_mixture_cut4.py:1329",
                "bayesfilter/highdim/filtering.py:scalar_nonlinear_fixed_design_tt_value_path",
            ],
            known_parameterization=sv_parameterization,
            known_observations=sv_observations,
            source_anchors=[
                *paper_algorithms,
                "paper Section 6.2",
                "third_party/audit/tensor-ssm-paper-demo/eg2_sv/mainscript.m:12",
                *author_recursion,
            ],
            blockers=[
                *common_posterior_blockers,
                {
                    "code": "BLOCK_SOURCE_ROUTE_MISMATCH",
                    "detail": (
                        "Current factorized scalar fixed-design fitter is not the author adaptive "
                        "TTSIRT retained-object recursion; wrapper documentation also disclaims it."
                    ),
                },
                {
                    "code": "MISSING_PRODUCTION_FIXED_ROUTE",
                    "detail": "No admitted source-route fixed-HMC value/score posterior adapter exists.",
                },
            ],
        ),
        _cell(
            cell_id="KSC-UKF",
            model_target="KSC finite Gaussian-mixture transformed stochastic volatility",
            filter_likelihood="component-enumerated principal-square-root UKF",
            route_classification="BAYESFILTER_APPROXIMATION",
            current_routes=[
                "bayesfilter/highdim/sv_mixture_cut4.py:2062",
                "bayesfilter/highdim/sv_mixture_cut4.py:2151",
            ],
            known_parameterization=sv_parameterization,
            known_observations={
                "candidate_fixture": "synthetic seed 81101, T=1000 in existing benchmark",
                "transform": "log(y^2 + 1e-8)",
                "mixture": "KSC 1998 seven-component table",
                "serious_data_status": "not frozen",
            },
            blockers=[
                *common_posterior_blockers,
                {
                    "code": "MISSING_UKF_REGION_ADMISSION",
                    "detail": "Square-root branch/SPD/floor behavior is not admitted over a posterior region.",
                },
                {
                    "code": "MISSING_BATCH_NATIVE_POSTERIOR_ADAPTER",
                    "detail": "Current component-enumerated filter is not a registered batched posterior adapter.",
                },
            ],
        ),
    ]
    pp_parameterization = {
        "dimension": 6,
        "physical_order": ["r", "K", "a", "s", "u", "v"],
        "parameter_box": [[0.1, 1.1], [110.0, 130.0], [20.0, 30.0], [0.1, 1.1], [0.0, 1.0], [0.0, 1.0]],
        "chart_status": "physical bounded coordinates only; HMC bijector/Jacobian not defined",
    }
    pp_observations = {
        "tiny_fixture": [[51.0, 4.6], [80.0, 3.8]],
        "candidate_serious_fixture": "synthetic seed 81104, T=20 in existing benchmark",
        "serious_data_status": "not frozen",
    }
    pp_common = [
        *common_posterior_blockers,
        {
            "code": "MISSING_UNCONSTRAINING_CHART_AND_JACOBIAN",
            "detail": "Bounded physical parameter box has no frozen HMC chart or log-Jacobian.",
        },
        {
            "code": "MISSING_BATCH_NATIVE_POSTERIOR_ADAPTER",
            "detail": "Existing test closures are likelihood components, not a registered posterior adapter.",
        },
    ]
    rows.extend(
        [
            _cell(
                cell_id="PP-SGQF",
                model_target="parameterized predator-prey state-space model",
                filter_likelihood="fixed SGQF structural approximation",
                route_classification="BAYESFILTER_APPROXIMATION",
                current_routes=[
                    "tests/highdim/test_p47_predator_prey_filtering.py:258",
                    "bayesfilter/nonlinear/fixed_sgqf_structural_adapter_tf.py",
                ],
                known_parameterization=pp_parameterization,
                known_observations=pp_observations,
                blockers=[
                    *pp_common,
                    {
                        "code": "MISSING_SERIOUS_SGQF_SETTINGS",
                        "detail": "Sparse design and branch thresholds are fixture settings, not frozen defaults.",
                    },
                ],
            ),
            _cell(
                cell_id="PP-UKF",
                model_target="parameterized predator-prey state-space model",
                filter_likelihood="structural principal-square-root UKF approximation",
                route_classification="BAYESFILTER_APPROXIMATION",
                current_routes=[
                    "tests/highdim/test_p47_predator_prey_filtering.py:214",
                    "tests/highdim/test_p47_predator_prey_filtering.py:275",
                ],
                known_parameterization=pp_parameterization,
                known_observations=pp_observations,
                blockers=[
                    *pp_common,
                    {
                        "code": "MISSING_UKF_REGION_ADMISSION",
                        "detail": "Principal-square-root/status behavior is not admitted over the posterior region.",
                    },
                ],
            ),
            _cell(
                cell_id="PP-ZC",
                model_target="parameterized predator-prey state-space model",
                filter_likelihood="current generic all-axes fixed-design TT diagnostic",
                route_classification="EXTENSION_OR_INVENTION_CURRENT_ROUTE",
                current_routes=[
                    "tests/highdim/test_p47_predator_prey_filtering.py:201",
                    "bayesfilter/highdim/filtering.py:1761",
                ],
                known_parameterization=pp_parameterization,
                known_observations=pp_observations,
                source_anchors=[
                    *paper_algorithms,
                    "paper Section 6.4 equation (38)",
                    "third_party/audit/tensor-ssm-paper-demo/eg4_predatorprey/mainscript.m:12",
                    *author_recursion,
                ],
                blockers=[
                    *pp_common,
                    {
                        "code": "BLOCK_SOURCE_ROUTE_MISMATCH",
                        "detail": "Generic all-axes retained grid is diagnostic/historical and not author TTSIRT recursion.",
                    },
                    {
                        "code": "MISSING_PRODUCTION_FIXED_ROUTE",
                        "detail": "No admitted source-route same-target value/score posterior binding exists.",
                    },
                ],
            ),
        ]
    )
    structural_parameters = {
        "candidate_parameters": ["rho", "sigma", "phi", "gamma", "R"],
        "inferred_subset": None,
        "chart_status": "not designed",
        "deterministic_identity": "k_t - phi*k_(t-1) - gamma*m_t^2 = 0",
    }
    structural_observations = {
        "worked_example_only": True,
        "graph_native_dataset": None,
        "serious_data_status": "missing",
    }
    structural_common = [
        *common_posterior_blockers,
        {
            "code": "MISSING_GRAPH_NATIVE_MODEL_AND_DATA",
            "detail": "Current worked fixture is NumPy/reference-only; no TF parameter posterior/data exists.",
        },
        {
            "code": "MISSING_PARAMETER_SUBSET_AND_CHART",
            "detail": "Inferred parameters, prior, support, chart, and Jacobian are not selected.",
        },
        {
            "code": "MISSING_STRUCTURAL_NEGATIVE_CONTROL",
            "detail": "Naive artificial-noise UKF negative control is not implemented as a distinct route.",
        },
    ]
    rows.extend(
        [
            _cell(
                cell_id="STR-UKF",
                model_target="Chapter 18b quadratic structural state-space model",
                filter_likelihood="structural UKF preserving deterministic k_t",
                route_classification="PLANNED_BAYESFILTER_APPROXIMATION",
                current_routes=[
                    "bayesfilter/testing/structural_fixtures.py:108",
                    "tests/test_structural_sigma_points.py:164",
                    "docs/chapters/ch18b_structural_deterministic_dynamics.tex:1011",
                ],
                known_parameterization=structural_parameters,
                known_observations=structural_observations,
                blockers=structural_common,
            ),
            _cell(
                cell_id="STR-ZC",
                model_target="Chapter 18b quadratic structural state-space model",
                filter_likelihood="planned fixed TT/SIRT-inspired extension",
                route_classification="EXTENSION_OR_INVENTION_BY_DEFINITION",
                current_routes=[],
                known_parameterization=structural_parameters,
                known_observations=structural_observations,
                source_anchors=paper_algorithms,
                blockers=[
                    *structural_common,
                    {
                        "code": "MISSING_EXTENSION_DESIGN",
                        "detail": "No graph-native Zhao-Cui-inspired extension value/score route exists.",
                    },
                ],
            ),
        ]
    )
    sir_parameters = {
        "dimension": 3,
        "unconstrained_order": ["log_kappa_scale", "log_nu_scale", "log_obs_noise_scale"],
        "physical_map": [
            "kappa=base_kappa*exp(theta[0])",
            "nu=base_nu*exp(theta[1])",
            "R=base_R*exp(2*theta[2])",
        ],
        "prior_status": "missing",
    }
    sir_observations = {
        "author_paper_scope": "state inference with fixed kappa/nu, J=9, T=20",
        "BayesFilter_scope": "three-parameter extension",
        "candidate_fixture": "synthetic Austria J=9, T=20",
        "serious_data_status": "not frozen",
    }
    sir_common = [
        *common_posterior_blockers,
        {
            "code": "MISSING_FULL_OBSERVED_DATA_FILTER_POSTERIOR",
            "detail": "Current parameter score evidence is local/complete-data or component scoped.",
        },
        {
            "code": "PAPER_TARGET_DIFFERS",
            "detail": "Paper Section 6.3 fixes kappa/nu and estimates state; parameterized SIR is an extension.",
        },
    ]
    rows.extend(
        [
            _cell(
                cell_id="SIR-SGQF",
                model_target="three-parameter spatial Austria SIR",
                filter_likelihood="planned fixed SGQF observed-data approximation",
                route_classification="PLANNED_BAYESFILTER_APPROXIMATION",
                current_routes=[],
                known_parameterization=sir_parameters,
                known_observations=sir_observations,
                blockers=[
                    *sir_common,
                    {
                        "code": "MISSING_SGQF_POSTERIOR_ROUTE",
                        "detail": "No admitted SGQF value/score observed-data parameter posterior exists.",
                    },
                ],
            ),
            _cell(
                cell_id="SIR-UKF",
                model_target="three-parameter spatial Austria SIR",
                filter_likelihood="planned structural UKF observed-data approximation",
                route_classification="BAYESFILTER_SCOUT_ONLY",
                current_routes=[
                    "tests/highdim/test_filtering_value_gradient_benchmark_deterministic_filters.py",
                ],
                known_parameterization=sir_parameters,
                known_observations=sir_observations,
                blockers=[
                    *sir_common,
                    {
                        "code": "UKF_SCOUT_NOT_POSTERIOR",
                        "detail": "Current UKF evidence is lower-rung/scout and not a registered full posterior.",
                    },
                ],
            ),
            _cell(
                cell_id="SIR-ZC",
                model_target="three-parameter spatial Austria SIR",
                filter_likelihood="planned fixed TTSIRT observed-data approximation",
                route_classification="FIXED_HMC_ADAPTATION_SUBSTRATE_INCOMPLETE_EXTENSION_TARGET",
                current_routes=[
                    "bayesfilter/highdim/models.py:935",
                    "bayesfilter/highdim/source_route.py",
                    "tests/highdim/test_p91_score_identity.py:45",
                ],
                known_parameterization=sir_parameters,
                known_observations=sir_observations,
                source_anchors=[
                    *paper_algorithms,
                    "paper Section 6.3 equation (37)",
                    "third_party/audit/tensor-ssm-paper-demo/eg3_sir/mainscript.m:12",
                    *author_recursion,
                ],
                blockers=[
                    *sir_common,
                    {
                        "code": "MISSING_RETAINED_OBJECT_VALUE_SCORE_POSTERIOR",
                        "detail": "Source-route substrate has no admitted full observed-data parameter posterior evaluator.",
                    },
                    {
                        "code": "PARAMETER_INFERENCE_IS_EXTENSION",
                        "detail": "Author SIR example has d=0 and does not infer kappa/nu/noise scales.",
                    },
                ],
            ),
        ]
    )
    return rows


def _assumptions() -> list[dict[str, Any]]:
    return [
        {
            "id": "A01",
            "choice": "LGSSM shared sequential HMC controller",
            "provenance": "completed LGSSM consolidation reset memo",
            "justification": "reusable controller mechanics",
            "failure_mode": "cell adapter could violate target identity or batching",
            "early_diagnostic": "P1 adapter identity and archive negative tests",
            "status": "REVIEWED_FOUNDATION_REVALIDATE_P1",
        },
        {
            "id": "A02",
            "choice": "plain dense IAF candidate arm",
            "provenance": "existing BayesFilter NeuTra trainer",
            "justification": "common baseline with frozen artifact schema",
            "failure_mode": "capacity/topology inadequate for nonlinear target",
            "early_diagnostic": "target-specific family screen",
            "status": "BASELINE_HYPOTHESIS",
        },
        {
            "id": "A03",
            "choice": "target-specific enhanced candidate arm",
            "provenance": "master-program repair after independent review",
            "justification": "prevents first-recipe failure from becoming cell rejection",
            "failure_mode": "family unavailable in current code or remains inadequate",
            "early_diagnostic": "P0/P1 capability inventory before model phase",
            "status": "REQUIRED_ARM_FAMILY_NOT_YET_FROZEN",
        },
        {
            "id": "A04",
            "choice": "fresh 5,000-step selected training per arm",
            "provenance": "LGSSM protocol and owner direction",
            "justification": "common serious training rung",
            "failure_mode": "undertraining or wasted compute on new target",
            "early_diagnostic": "screen/heldout/downstream nomination before final run",
            "status": "COMMON_RUNG_NOT_ADEQUACY_PROOF",
        },
        {
            "id": "A05",
            "choice": "posterior equivalence margins",
            "provenance": "program evidence contract",
            "justification": "same-target sampler agreement needs practical regions",
            "failure_mode": "post-result or arbitrary thresholds",
            "early_diagnostic": "model-phase statistical design before serious output",
            "status": "NOT_FROZEN_TARGET_BLOCKER",
        },
        {
            "id": "A06",
            "choice": "synthetic benchmark observations as candidate serious data",
            "provenance": "existing benchmarks seeds 81101 and 81104",
            "justification": "reproducible starting candidates",
            "failure_mode": "benchmark data silently becomes scientific default",
            "early_diagnostic": "freeze data choice/hash in owning model phase",
            "status": "CANDIDATE_ONLY",
        },
        {
            "id": "A07",
            "choice": "existing filter settings",
            "provenance": "current unit tests and historical campaigns",
            "justification": "useful lower-rung starting values",
            "failure_mode": "fixture settings fail over posterior region",
            "early_diagnostic": "target-region filter admission ladder",
            "status": "WARM_START_HYPOTHESES",
        },
        {
            "id": "A08",
            "choice": "Zhao-Cui current wrapper classifications",
            "provenance": "checked JMLR paper, author source, and local p56 audit",
            "justification": "prevents generic/fixed grid from being called source-faithful",
            "failure_mode": "component anchor promoted to whole-route fidelity",
            "early_diagnostic": "operation-by-operation source ledger in owning phase",
            "status": "REVIEWED_P0_CLASSIFICATION",
        },
        {
            "id": "A09",
            "choice": "GPU/XLA training with memory growth",
            "provenance": "repository execution policy",
            "justification": "required learned-workload target",
            "failure_mode": "CPU or non-XLA evidence mislabeled serious",
            "early_diagnostic": "P1 trusted device canary",
            "status": "REQUIRED",
        },
        {
            "id": "A10",
            "choice": "P0 target signatures withheld",
            "provenance": "posterior identity gate",
            "justification": "unknown priors/data/charts cannot be hashed into an admitted target",
            "failure_mode": "scope hash misused as posterior identity",
            "early_diagnostic": "schema rejects non-null target signature before complete fields",
            "status": "REQUIRED_FAIL_CLOSED",
        },
    ]


def _commands(output_root: str, generated_at: str) -> list[dict[str, Any]]:
    return [
        {
            "phase": "P0",
            "status": "EXECUTED_BY_THIS_BUILDER",
            "command": (
                "python docs/benchmarks/build_multimodel_neutra_p0_registry.py "
                f"--output-root {output_root} --generated-at {generated_at}"
            ),
            "environment": "current repository Python; CPU-only metadata work; no framework import",
            "device_intent": "CPU_ONLY_NO_TENSORFLOW_IMPORT",
            "timeout_seconds": 300,
            "expected_artifacts": [
                "target_registry.json",
                "cell_ledger.json",
                "assumption_ledger.json",
                "command_manifest.json",
                "budget_ledger.json",
                "execution_events.jsonl",
                "run_manifest.json",
            ],
        },
        {
            "phase": "P1",
            "status": "TO_BE_CREATED_IN_PHASE",
            "command": None,
            "environment": "project TensorFlow/TFP environment",
            "device_intent": "CPU_REFERENCE_PLUS_TRUSTED_GPU_XLA_CANARY",
            "timeout_seconds": None,
            "expected_artifacts": ["generic harness tests", "trusted GPU/XLA canary"],
            "blocker": "P1 implementation and exact focused-test commands do not yet exist",
        },
        *[
            {
                "phase": phase,
                "status": "TO_BE_CREATED_IN_PHASE",
                "command": None,
                "environment": "project TensorFlow/TFP environment",
                "device_intent": "TRUSTED_GPU_XLA_FOR_SERIOUS_RUNS",
                "timeout_seconds": None,
                "expected_artifacts": ["cell-specific artifacts named by subplan"],
                "blocker": "all owning cells are TARGET_BLOCKED at P0",
            }
            for phase in ("P2", "P3", "P4", "P5", "P6")
        ],
        {
            "phase": "P7",
            "status": "TO_BE_REFRESHED_AFTER_P6",
            "command": None,
            "environment": "current repository Python",
            "device_intent": "CPU_ONLY_SYNTHESIS",
            "timeout_seconds": None,
            "expected_artifacts": ["terminal integrity audit and reset memo"],
            "blocker": "requires P0-P6 results",
        },
    ]


def _budget() -> dict[str, Any]:
    phases = {
        "P0": {"cpu_hours": 8, "gpu_hours": 0},
        "P1": {"cpu_hours": 16, "gpu_hours": 2},
        "P2": {"cpu_hours": 16, "gpu_hours": 80},
        "P3": {"cpu_hours": 8, "gpu_hours": 40},
        "P4": {"cpu_hours": 24, "gpu_hours": 120},
        "P5": {"cpu_hours": 24, "gpu_hours": 80},
        "P6": {"cpu_hours": 32, "gpu_hours": 120},
        "P7": {"cpu_hours": 8, "gpu_hours": 0},
    }
    return {
        "schema": f"{SCHEMA_VERSION}.budget_ledger",
        "program_id": PROGRAM_ID,
        "phase_ceilings": phases,
        "program_ceiling": {
            "cpu_hours": sum(item["cpu_hours"] for item in phases.values()),
            "gpu_hours": sum(item["gpu_hours"] for item in phases.values()),
        },
        "per_cell_gpu_buckets": {
            "plain_dense_iaf_arm": 15,
            "target_specific_enhanced_arm": 15,
            "plain_hmc_comparator": 6,
            "cell_admission_and_cell_specific_artifacts": 4,
            "total": 40,
        },
        "shared_p1_gpu_bucket": 2,
        "consumed": {
            "P0_cpu_hours": 0,
            "P0_gpu_hours": 0,
            "measurement": "wall time is recorded in run manifest; planning work below one hour",
        },
        "rule": (
            "unexecuted mandatory candidate arm or exhausted unanswered bucket is a blocker, "
            "not CELL_CANDIDATE_REJECTED"
        ),
    }


def _scholarly_ledgers() -> dict[str, Any]:
    paper_path = ".localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.pdf"
    author_root = "third_party/audit/tensor-ssm-paper-demo"
    source_support = {
        "schema": f"{SCHEMA_VERSION}.source_support",
        "program_id": PROGRAM_ID,
        "sources": [
            {
                "source_id": "zhao_cui_jmlr_2024",
                "classification": "DIRECT_METHOD",
                "title": "Tensor-Train Methods for Sequential State and Parameter Learning in State-Space Models",
                "authors": ["Yiran Zhao", "Tiangang Cui"],
                "year": 2024,
                "venue": "Journal of Machine Learning Research 25",
                "pages": "1-51",
                "local_full_text": paper_path,
                "publication_status": "published",
                "license": "CC-BY 4.0 stated in paper front matter",
                "retraction_status": "NOT_CHECKED_AGAINST_LIVE_INDEX; no notice in local published full text",
                "inspected_technical_anchors": [
                    "Section 2.1 equations (9)-(11)",
                    "Algorithm 1 equation (12)",
                    "Proposition 2",
                    "Algorithms 2-5",
                    "Section 5 preconditioning",
                    "Sections 6.2-6.4",
                ],
                "allowed_claims": [
                    "author method recursively approximates joint state/parameter posteriors using TT reapproximation and marginalization",
                    "squared-TT approximations define KR maps and retained sequential objects",
                    "paper examples include SV, fixed-parameter state-only SIR, and parameterized predator-prey",
                ],
                "forbidden_claims": [
                    "BayesFilter generic retained grid is source-faithful",
                    "paper establishes HMC differentiation of frozen fixed branches",
                    "paper SIR example estimates kappa, nu, or observation-noise parameters",
                    "paper covers the Chapter 18b structural model",
                ],
            },
            {
                "source_id": "zhao_cui_author_code",
                "classification": "IMPLEMENTATION_OR_SOFTWARE",
                "title": "tensor-ssm-paper-demo author source snapshot",
                "local_root": author_root,
                "upstream_repository": "https://github.com/DeepTransport/tensor-ssm-paper-demo",
                "upstream_commit": "80034dccb99eb1d86284a1839b4a12067d13b9da",
                "snapshot_manifest": "third_party/audit/zhao_cui_tensor_ssm_p10/MANIFEST.yml",
                "license_boundary": (
                    "audit-only snapshot; do not import into production without separate license and clean-room decision"
                ),
                "inspected_technical_anchors": [
                    "models/full_sol.m:21-134",
                    "models/pre_sol.m:16-213",
                    "deep-tensor.dev/src/@TTSIRT/marginalise.m:1-85",
                    "deep-tensor.dev/src/@TTFun/cross.m:8-169",
                    "eg2_sv/mainscript.m:12-58",
                    "eg3_sir/mainscript.m:12-64",
                    "eg4_predatorprey/mainscript.m:12-94",
                ],
                "allowed_claims": [
                    "author code uses adaptive/random TT construction, retained SIRTs, marginalization, recentering, and sequential recursion",
                    "author SIR script sets parameter dimension d=0",
                    "author predator-prey script sets parameter dimension d=6",
                ],
                "forbidden_claims": [
                    "author code is production BayesFilter code",
                    "source snapshot proves BayesFilter implementation correctness",
                    "author source may be copied into production without boundary review",
                ],
            },
        ],
    }
    citation_metadata = {
        "schema": f"{SCHEMA_VERSION}.citation_venue_metadata",
        "program_id": PROGRAM_ID,
        "metadata_date": "2026-07-15",
        "records": [
            {
                "source_id": "zhao_cui_jmlr_2024",
                "venue": "Journal of Machine Learning Research",
                "citation_count": None,
                "citation_source": None,
                "venue_rank": None,
                "status": "LIVE_METADATA_NOT_QUERIED_NOT_NEEDED_FOR_IMPLEMENTATION_GATE",
                "caveat": "citation/venue metrics are coverage signals, not correctness evidence",
            },
            {
                "source_id": "cached_openalex_file_named_for_zhao_cui",
                "path": (
                    ".complete-highdim-source-snapshot-complete-highdim-leaderboard-20260711-221500/"
                    ".local_sources/highdim_nonlinear_filtering/openalex_zhao_cui_jmlr_2024.json"
                ),
                "status": "QUARANTINED_MISLABELED_CONTENT",
                "reason": "inspected payload contains unrelated computer-vision records, not reliable Zhao-Cui metadata",
            },
        ],
    }
    backward = {
        "schema": f"{SCHEMA_VERSION}.backward_snowball",
        "program_id": PROGRAM_ID,
        "seed_source": "zhao_cui_jmlr_2024",
        "records": [
            {
                "work": "Cui and Dolgov (2022), Deep composition of tensor-trains using squared inverse Rosenblatt transports",
                "classification": "FOUNDATIONAL",
                "action": "inspect in owning Zhao-Cui implementation phase before new Proposition-2/KR claims",
            },
            {
                "work": "Spantini et al. (2018), inference via low-dimensional couplings",
                "classification": "FOUNDATIONAL",
                "action": "context only for present P0; inspect before decomposition-theorem claims",
            },
            {
                "work": "Griebel and Harbrecht (2023), TT approximation error source used by Proposition 9",
                "classification": "FOUNDATIONAL",
                "action": "not needed for route identity; inspect before approximation-rate claims",
            },
            {
                "work": "Reich (2013), ensemble transform particle filtering",
                "classification": "COMPETITOR",
                "action": "outside P0 route-identity scope; retain as future comparison risk",
            },
            {
                "work": "Spantini et al. (2022), ensemble transport smoothing",
                "classification": "COMPETITOR",
                "action": "outside P0 route-identity scope; retain as future comparison risk",
            },
        ],
    }
    forward = {
        "schema": f"{SCHEMA_VERSION}.forward_snowball",
        "program_id": PROGRAM_ID,
        "seed_source": "zhao_cui_jmlr_2024",
        "query_date": "2026-07-15",
        "query_source": None,
        "status": "NOT_QUERIED_NETWORK_NOT_NEEDED_FOR_P0_IMPLEMENTATION_GATE",
        "blocker": "No trustworthy local citing-work index; mislabeled OpenAlex cache quarantined.",
        "impact": "does not block checked paper/author-source route classification",
    }
    claim_support = {
        "schema": f"{SCHEMA_VERSION}.claim_support",
        "program_id": PROGRAM_ID,
        "claims": [
            {
                "claim": "Zhao-Cui recursion retains and marginalizes a joint state/parameter approximation",
                "support_class": "PRIMARY_TECHNICAL_SUPPORT_PLUS_IMPLEMENTATION_EVIDENCE",
                "anchors": ["paper equations (9)-(12), Algorithm 1, Proposition 2", "author full_sol.m:21-134"],
            },
            {
                "claim": "Current SVX-ZC scalar wrapper is not the author coupled/adaptive TTSIRT recursion",
                "support_class": "IMPLEMENTATION_EVIDENCE",
                "anchors": ["bayesfilter/highdim/sv_mixture_cut4.py:1267-1324", "author full_sol.m:21-129"],
            },
            {
                "claim": "Current PP-ZC generic all-axes retained grid is not production source-route Zhao-Cui",
                "support_class": "IMPLEMENTATION_EVIDENCE",
                "anchors": ["tests/highdim/test_p47_predator_prey_filtering.py:201-211", "bayesfilter/highdim/filtering.py:1761", "author full_sol.m:21-129"],
            },
            {
                "claim": "Parameterized SIR is an extension beyond the paper's SIR example",
                "support_class": "PRIMARY_TECHNICAL_SUPPORT_PLUS_IMPLEMENTATION_EVIDENCE",
                "anchors": ["paper Section 6.3 states fixed kappa/nu", "author eg3_sir/mainscript.m:14", "bayesfilter/highdim/models.py:935-1005"],
            },
            {
                "claim": "Chapter 18b Zhao-Cui application is invention",
                "support_class": "PROJECT_SCOPE_CLASSIFICATION",
                "anchors": ["docs/chapters/ch18b_structural_deterministic_dynamics.tex:1011", "paper examples 6.1-6.4 omit this model"],
            },
        ],
    }
    omissions = {
        "schema": f"{SCHEMA_VERSION}.omitted_paper_risks",
        "program_id": PROGRAM_ID,
        "records": [
            {
                "candidate": "Cui and Dolgov (2022)",
                "risk": "load-bearing Proposition-2/KR foundation not re-audited in this P0",
                "reason": "P0 classifies existing routes; it does not implement new KR mathematics",
                "next_action": "mandatory technical inspection before P2/P4/P6 source-route implementation claims",
            },
            {
                "candidate": "recent TT filtering extensions and replications",
                "risk": "forward-snowball coverage unavailable",
                "reason": "no trustworthy local citing index and network lookup not needed for current gate",
                "next_action": "query a trustworthy index before publication-grade literature claims",
            },
            {
                "candidate": "filter-specific SGQF and UKF primary sources",
                "risk": "not audited in Zhao-Cui-focused P0 source ledger",
                "reason": "P0 does not make source-faithfulness claims for those BayesFilter approximations",
                "next_action": "inspect in P2-P6 when filter mathematics is materially changed or promoted",
            },
        ],
    }
    return {
        "source_support.json": source_support,
        "citation_venue_metadata.json": citation_metadata,
        "backward_snowball.json": backward,
        "forward_snowball.json": forward,
        "claim_support.json": claim_support,
        "omitted_paper_risks.json": omissions,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--generated-at", default=datetime.now().astimezone().isoformat(timespec="seconds"))
    args = parser.parse_args()

    repository = Path.cwd()
    missing = [path for path in SOURCE_PATHS if not (repository / path).is_file()]
    if missing:
        raise SystemExit(f"missing required P0 sources: {missing}")

    output_root = args.output_root
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    cells = _cells()
    if len(cells) != 11 or len({item["cell_id"] for item in cells}) != 11:
        raise SystemExit("P0 registry must contain exactly eleven unique cells")
    if any(item["target_signature"] is not None for item in cells):
        raise SystemExit("P0 must not issue posterior signatures for blocked cells")

    source_hashes = {
        path: _file_hash(repository / path)
        for path in SOURCE_PATHS
    }
    target_registry = {
        "schema": f"{SCHEMA_VERSION}.target_registry",
        "program_id": PROGRAM_ID,
        "generated_at": args.generated_at,
        "registry_status": "NO_POSTERIOR_SIGNATURES_ISSUED_ALL_CELLS_TARGET_BLOCKED",
        "signature_issuance_rule": (
            "a non-null target signature requires frozen prior, data hash, parameter chart and "
            "Jacobian, filter settings/dependencies, and independent posterior recomposition"
        ),
        "cells": cells,
        "source_hashes": source_hashes,
    }
    cell_ledger = {
        "schema": f"{SCHEMA_VERSION}.cell_ledger",
        "program_id": PROGRAM_ID,
        "generated_at": args.generated_at,
        "current_states": {item["cell_id"]: item["state"] for item in cells},
        "transition_history": [
            {
                "cell_id": item["cell_id"],
                "from": "UNINVENTORIED",
                "to": "TARGET_BLOCKED",
                "reason_codes": [entry["code"] for entry in item["blockers"]],
                "evidence": "target_registry.json",
            }
            for item in cells
        ],
    }
    assumption_ledger = {
        "schema": f"{SCHEMA_VERSION}.assumption_ledger",
        "program_id": PROGRAM_ID,
        "generated_at": args.generated_at,
        "assumptions": _assumptions(),
    }
    command_manifest = {
        "schema": f"{SCHEMA_VERSION}.command_manifest",
        "program_id": PROGRAM_ID,
        "generated_at": args.generated_at,
        "commands": _commands(str(output_root), args.generated_at),
    }
    budget_ledger = _budget()
    scholarly = _scholarly_ledgers()
    run_manifest = {
        "schema": f"{SCHEMA_VERSION}.run_manifest",
        "program_id": PROGRAM_ID,
        "phase": "P0",
        "attempt": "attempt-01",
        "generated_at": args.generated_at,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_dirty_entry_count": len(_git("status", "--porcelain=v1").splitlines()),
        "command": (
            "python docs/benchmarks/build_multimodel_neutra_p0_registry.py "
            f"--output-root {output_root} --generated-at {args.generated_at}"
        ),
        "environment": "current repository Python",
        "cpu_gpu_status": "CPU_METADATA_ONLY_NO_TENSORFLOW_IMPORT_NO_GPU_USE",
        "data_version": "N/A_NO_EXPERIMENT_DATA_EXECUTED",
        "random_seeds": "N/A_DETERMINISTIC_METADATA_BUILD",
        "wall_time_seconds": time.monotonic() - started,
        "output_root": str(output_root),
        "plan_file": PLAN_PATH,
        "result_file": RESULT_PATH,
        "source_hashes": source_hashes,
        "dirty_worktree_disclosure": (
            "Repository contains extensive pre-existing concurrent-lane changes; this P0 build "
            "creates only uniquely named program artifacts."
        ),
    }
    events = [
        {
            "event": "P0_LAUNCH",
            "at": args.generated_at,
            "attempt": "attempt-01",
            "device_intent": "CPU_METADATA_ONLY",
        },
        {
            "event": "P0_CELL_CLASSIFICATION",
            "at": args.generated_at,
            "counts": {"TARGET_BLOCKED": 11},
            "posterior_signatures_issued": 0,
        },
    ]

    payloads = {
        "target_registry.json": target_registry,
        "cell_ledger.json": cell_ledger,
        "assumption_ledger.json": assumption_ledger,
        "command_manifest.json": command_manifest,
        "budget_ledger.json": budget_ledger,
        "run_manifest.json": run_manifest,
        **scholarly,
    }
    for name, payload in payloads.items():
        _write_json(output_root / name, payload)
    (output_root / "execution_events.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in events),
        encoding="utf-8",
    )

    artifact_hashes = {
        "schema": f"{SCHEMA_VERSION}.artifact_hashes",
        "program_id": PROGRAM_ID,
        "artifacts": {
            name: _file_hash(output_root / name)
            for name in (*payloads.keys(), "execution_events.jsonl")
        },
    }
    _write_json(output_root / "artifact_hashes.json", artifact_hashes)
    print(json.dumps({
        "output_root": str(output_root),
        "cell_count": len(cells),
        "target_blocked_count": sum(item["state"] == "TARGET_BLOCKED" for item in cells),
        "target_signatures_issued": sum(item["target_signature"] is not None for item in cells),
        "program_gpu_hours": budget_ledger["program_ceiling"]["gpu_hours"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
