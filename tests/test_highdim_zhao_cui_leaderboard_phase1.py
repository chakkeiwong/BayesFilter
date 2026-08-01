from __future__ import annotations

import importlib.util
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_two_lane_highdim_leaderboard_phase1", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load highdim leaderboard module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase1_zhao_cui_sv_and_ksc_rows_use_fixed_variant_manual_scores(monkeypatch) -> None:
    module = _load_module()
    full_observations = module._sv_observations()
    original_config = module._zhao_cui_scalar_tt_config
    monkeypatch.setattr(module, "_sv_observations", lambda: full_observations[:4])
    monkeypatch.setattr(
        module,
        "_zhao_cui_scalar_tt_config",
        lambda seed: original_config(
            seed,
            basis_degree=16,
            quadrature_order=41,
        ),
    )

    rows = [module._zhao_cui_actual_sv_tt_cell(), module._zhao_cui_ksc_tt_cell()]

    for row in rows:
        assert row["algorithm_id"] == "zhao_cui_scalar_or_multistate"
        assert row["comparison_status"] == "executed_value_score"
        assert row["route_role"] == "fixed_variant_analytical_hmc_route"
        assert row["hmc_target_scope_admitted"] is True
        assert row["score_status"] == "analytical_score_emitted"
        assert row["score_coordinate_system"] == "theta=[probit_gamma,log_beta]"
        assert isinstance(row["score"], list)
        assert row["score"]
        assert all(tf.math.is_finite(tf.constant(row["score"], dtype=tf.float64)).numpy())
        provenance = str(row["score_derivative_provenance"]).lower()
        assert "manual_parameter_score_methods_only" in provenance
        assert "autodiff" not in provenance
        assert "gradienttape" not in provenance
        assert "gradient_tape" not in provenance
        assert "finite_difference" not in provenance
        assert "fd_" not in provenance


def test_phase1_zhao_cui_derivative_config_disables_fd() -> None:
    module = _load_module()
    config = module._zhao_cui_derivative_config_no_fd()

    assert config.parameter_indices == (0, 1)
    assert config.finite_difference_h == ()


def test_phase1_enforcement_admits_only_repaired_manual_zhao_cui_rows() -> None:
    module = _load_module()
    repaired = {
        "row_id": "zhao_cui_sv_actual_nongaussian_T1000",
        "algorithm_id": "zhao_cui_scalar_or_multistate",
        "comparison_status": "executed_value_score",
        "numeric_execution_status": "executed_zhao_cui_exact_transformed_sv_fixed_branch_tt_value_score",
        "score": [1.0, -0.25],
        "score_l2_norm": 1.0307764064,
        "score_coordinate_system": "theta=[probit_gamma,log_beta]",
        "score_derivative_provenance": (
            "zhao_cui_scalar_fixed_branch_tt_exact_transformed_sv_manual_parameter_score_methods_only"
        ),
        "route_id": "zhao_cui_exact_transformed_sv_fixed_branch_tt",
        "route_role": "fixed_variant_analytical_hmc_route",
        "hmc_route_policy_id": "zhao_cui_fixed_variant_hmc_default_v1",
        "hmc_target_scope_admitted": True,
        "nonclaims": [],
    }
    autodiff = {
        **repaired,
        "score_derivative_provenance": "zhao_cui_scalar_fixed_branch_tt_gradient_tape_autodiff",
    }

    repaired_row, demoted_row = module._enforce_analytical_score_admission([repaired, autodiff])

    assert repaired_row["comparison_status"] == "executed_value_score"
    assert repaired_row["score"] == [1.0, -0.25]
    assert demoted_row["comparison_status"] == "executed_value_only"
    assert demoted_row["score_status"] == "blocked_autodiff_not_admitted"
    assert demoted_row["score"] is None
