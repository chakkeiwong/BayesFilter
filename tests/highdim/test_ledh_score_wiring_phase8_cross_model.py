from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from bayesfilter.highdim.ledh_forward_contract import (
    ACTUAL_SV_ROW_ID,
    FIXED_SIR_AUSTRIA_ROW_ID,
    GENERALIZED_SV_ROW_ID,
    KSC_SV_ROW_ID,
    LGSSM_M3_T50_ROW_ID,
    PREDATOR_PREY_ROW_ID,
    LEDH_FORWARD_ADMISSION_STATUS_HISTORICAL_RAW,
    validate_ledh_forward_scalar_artifact,
)
from bayesfilter.highdim.ledh_score_contract import (
    LEDH_SCORE_COMPACT_ACTUAL_SV_PROVENANCE,
    LEDH_SCORE_COMPACT_FIXED_SIR_PROVENANCE,
    LEDH_SCORE_COMPACT_GENERALIZED_SV_PROVENANCE,
    LEDH_SCORE_COMPACT_KSC_SV_PROVENANCE,
    LEDH_SCORE_COMPACT_LGSSM_PROVENANCE,
    LEDH_SCORE_COMPACT_PREDATOR_PREY_PROVENANCE,
)
from docs.benchmarks import benchmark_ledh_same_target_actual_sv_score as actual_sv
from docs.benchmarks import benchmark_ledh_same_target_fixed_sir_score as fixed_sir
from docs.benchmarks import benchmark_ledh_same_target_generalized_sv_score as generalized_sv
from docs.benchmarks import benchmark_ledh_same_target_ksc_sv_score as ksc_sv
from docs.benchmarks import benchmark_ledh_same_target_lgssm_m3_t50_value as lgssm
from docs.benchmarks import benchmark_ledh_same_target_predator_prey_score as predator_prey


ROOT = Path(__file__).resolve().parents[2]

ROW_CASES = (
    (
        LGSSM_M3_T50_ROW_ID,
        "docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json",
        "lgssm_gaussian_observation_density",
        "physical_benchmark_exact_oracle",
        ("phi1", "phi2", "phi3", "q_scale", "r_scale"),
        lgssm.COMPACT_SCORE_ROUTE_ID,
        LEDH_SCORE_COMPACT_LGSSM_PROVENANCE,
    ),
    (
        FIXED_SIR_AUSTRIA_ROW_ID,
        "docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json",
        "fixed_sir_infectious_components_gaussian_observation_density",
        "sir_log_scale_theta",
        ("log_kappa_scale", "log_nu_scale", "log_obs_noise_scale"),
        fixed_sir.FIXED_SIR_COMPACT_SCORE_ROUTE_ID,
        LEDH_SCORE_COMPACT_FIXED_SIR_PROVENANCE,
    ),
    (
        PREDATOR_PREY_ROW_ID,
        "docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json",
        "additive_gaussian_predator_prey",
        "physical",
        ("r", "K", "a", "s", "u", "v"),
        predator_prey.PREDATOR_PREY_COMPACT_SCORE_ROUTE_ID,
        LEDH_SCORE_COMPACT_PREDATOR_PREY_PROVENANCE,
    ),
    (
        ACTUAL_SV_ROW_ID,
        "docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json",
        "transformed_actual_sv_log_y_square",
        "synthetic_unconstrained",
        ("gamma_unconstrained", "log_beta"),
        actual_sv.ACTUAL_SV_COMPACT_SCORE_ROUTE_ID,
        LEDH_SCORE_COMPACT_ACTUAL_SV_PROVENANCE,
    ),
    (
        GENERALIZED_SV_ROW_ID,
        "docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json",
        "source_route_prior_mean_generalized_sv",
        "source_route_active_transformed_prior_mean",
        ("gamma_unconstrained", "log_tau", "mu"),
        generalized_sv.GENERALIZED_SV_COMPACT_SCORE_ROUTE_ID,
        LEDH_SCORE_COMPACT_GENERALIZED_SV_PROVENANCE,
    ),
    (
        KSC_SV_ROW_ID,
        "docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json",
        "ksc_log_chi_square_gaussian_mixture_surrogate",
        "synthetic_unconstrained",
        ("gamma_unconstrained", "log_beta"),
        ksc_sv.KSC_SV_COMPACT_SCORE_ROUTE_ID,
        LEDH_SCORE_COMPACT_KSC_SV_PROVENANCE,
    ),
)


@pytest.mark.parametrize(
    (
        "row_id",
        "artifact_path",
        "target_policy",
        "theta_system",
        "parameter_order",
        "adapter_provenance",
        "contract_provenance",
    ),
    ROW_CASES,
)
def test_phase8_row_contracts_match_compact_adapter_identity(
    row_id: str,
    artifact_path: str,
    target_policy: str,
    theta_system: str,
    parameter_order: tuple[str, ...],
    adapter_provenance: str,
    contract_provenance: str,
) -> None:
    payload = json.loads((ROOT / artifact_path).read_text(encoding="utf-8"))
    normalized = validate_ledh_forward_scalar_artifact(
        payload,
        expected_row_id=row_id,
        require_admitted=False,
    )

    assert normalized["target_observation_policy"] == target_policy
    assert normalized["theta_coordinate_system"] == theta_system
    assert tuple(normalized["forward_contract"]["theta_contract"]["parameter_order"]) == parameter_order
    assert normalized["admission_status"] == LEDH_FORWARD_ADMISSION_STATUS_HISTORICAL_RAW
    assert adapter_provenance == contract_provenance


@pytest.mark.parametrize(
    "module",
    (predator_prey, actual_sv, generalized_sv, ksc_sv),
)
def test_phase8_standalone_score_clis_default_to_production_precision(module) -> None:
    source = inspect.getsource(module._parse_args)  # noqa: SLF001

    assert 'parser.add_argument("--dtype", choices=("float64", "float32"), default="float32")' in source
    assert (
        'parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="enabled")'
        in source
    )


def test_phase8_lgssm_and_fixed_sir_inherit_production_precision_defaults() -> None:
    lgssm_source = inspect.getsource(lgssm._parse_args)  # noqa: SLF001
    fixed_sir_source = inspect.getsource(fixed_sir.p8p._parse_args)  # noqa: SLF001

    for source in (lgssm_source, fixed_sir_source):
        assert 'parser.add_argument("--dtype", choices=("float64", "float32"), default="float32")' in source
        assert 'default="enabled"' in source


@pytest.mark.parametrize(
    ("diagnostic", "compact_symbol", "historical_symbols", "value_symbol"),
    (
        (
            fixed_sir._fixed_sir_compact_coordinate_fd_diagnostic,  # noqa: SLF001
            "_compact_value_and_score_across_seeds",
            ("_fixed_sir_manual_score_diagnostic",),
            "_value_objective_across_seeds",
        ),
        (
            predator_prey._coordinate_fd_score_diagnostic,  # noqa: SLF001
            "_compact_value_and_score_across_seeds",
            ("_manual_value_and_score_across_seeds", "_manual_value_and_score_from_components"),
            "_value_objective_across_seeds",
        ),
        (
            actual_sv._coordinate_fd_score_diagnostic,  # noqa: SLF001
            "_compact_value_and_score_across_seeds",
            ("_manual_value_and_score_across_seeds", "_manual_value_and_score_from_components"),
            "_value_objective_across_seeds",
        ),
        (
            generalized_sv._coordinate_fd_score_diagnostic,  # noqa: SLF001
            "_compact_value_and_score_across_seeds",
            ("_manual_value_and_score_across_seeds",),
            "_value_objective_across_seeds",
        ),
        (
            ksc_sv._coordinate_fd_score_diagnostic,  # noqa: SLF001
            "_compact_value_and_score_across_seeds",
            ("_manual_value_and_score_across_seeds",),
            "_value_objective_across_seeds",
        ),
    ),
)
def test_phase8_default_diagnostics_use_compact_score_and_value_only_fd(
    diagnostic,
    compact_symbol: str,
    historical_symbols: tuple[str, ...],
    value_symbol: str,
) -> None:
    source = inspect.getsource(diagnostic)

    assert compact_symbol in source
    assert value_symbol in source
    assert all(symbol not in source for symbol in historical_symbols)


def test_phase8_lgssm_compact_dispatch_and_value_only_fd_are_separate() -> None:
    score_source = inspect.getsource(lgssm._score_only_diagnostic_from_tensors)  # noqa: SLF001
    fd_source = inspect.getsource(lgssm._fd_only_diagnostic_from_tensors)  # noqa: SLF001

    assert 'if args.score_mode == "manual-reverse"' in score_source
    assert "_compact_value_and_score_from_components" in score_source
    assert "score_mode = \"compact-sensitivity\"" in score_source
    assert "_same_target_value_from_components" in fd_source
    assert "_compact_value_and_score_from_components" not in fd_source


@pytest.mark.parametrize(
    ("module", "diagnostic"),
    (
        (fixed_sir, fixed_sir._fixed_sir_compact_coordinate_fd_diagnostic),  # noqa: SLF001
        (predator_prey, predator_prey._coordinate_fd_score_diagnostic),  # noqa: SLF001
        (actual_sv, actual_sv._coordinate_fd_score_diagnostic),  # noqa: SLF001
        (generalized_sv, generalized_sv._coordinate_fd_score_diagnostic),  # noqa: SLF001
        (ksc_sv, ksc_sv._coordinate_fd_score_diagnostic),  # noqa: SLF001
    ),
)
def test_phase8_nonlinear_rows_preserve_sequential_seed_schedule(module, diagnostic) -> None:
    wrapper_source = inspect.getsource(module._compact_value_and_score_across_seeds)  # noqa: SLF001
    value_source = inspect.getsource(module._value_objective_across_seeds)  # noqa: SLF001
    diagnostic_source = inspect.getsource(diagnostic)

    assert "for seed in args.batch_seeds" in wrapper_source
    assert "_single_seed_args(args, seed)" in wrapper_source
    assert "_compact_value_and_score_from_components" in wrapper_source
    assert "_compact_value_and_score_across_seeds" in diagnostic_source
    assert "for seed in args.batch_seeds" in value_source
    assert "_single_seed_args(args, seed)" in value_source
    assert "_value_objective_across_seeds" in diagnostic_source


def test_phase8_actual_and_ksc_targets_remain_distinct() -> None:
    assert ROW_CASES[3][2] == "transformed_actual_sv_log_y_square"
    assert ROW_CASES[5][2] == "ksc_log_chi_square_gaussian_mixture_surrogate"
    assert ROW_CASES[3][2] != ROW_CASES[5][2]

    ksc_artifact_source = inspect.getsource(ksc_sv._score_artifact_from_diagnostic)  # noqa: SLF001
    assert '"claims_exact_native_actual_sv_likelihood": False' in ksc_artifact_source


def test_phase8_cross_model_tests_do_not_construct_runtime_full_admission() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden_tokens = (
        "require_all_parameter_correctness" + "=True",
        "n10000_" + "memory_pass",
        "trusted_gpu_" + "score_memory_artifact",
    )

    assert all(token not in source for token in forbidden_tokens)
