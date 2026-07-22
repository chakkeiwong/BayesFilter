"""Contract tests for the repaired HNN-NeuTra comparison."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import tensorflow as tf

from bayesfilter.testing import hnn_neutra_exact_comparison_tf as comparison


def _diagnostic(ess: tuple[float, ...]):
    return {
        "parameter_diagnostics": tuple(
            {"parameter": f"p{index}", "bulk_ess": value}
            for index, value in enumerate(ess)
        )
    }


def test_primary_comparison_excludes_zero_residual_and_requires_exact_gradient():
    assert comparison.PRIMARY_ARMS == ("learned_residual", "true_gradient")
    source = inspect.getsource(comparison.force_arms)
    assert '"true_gradient"' in source
    assert '"zero_residual"' not in source


def test_structural_repair_is_exact_sampling_only_and_reuses_hnn_for_timing():
    source = inspect.getsource(comparison.run_structural_exact_repair)
    assert "train_fresh_force" not in source
    assert 'arm_id="true_gradient"' in source
    assert 'step_size=0.1' in source
    assert 'num_leapfrog_steps=8' in source
    assert "read_tensor_archive" in source
    assert "load_frozen_scalar_residual_force" in source
    assert "matched_mechanics_benchmark" in source


def test_performance_promotion_requires_matched_health_gate():
    source = inspect.getsource(comparison.run_full_cell)
    assert 'and matched["passed"]' in source


def test_direct_posterior_agreement_passes_identical_draws():
    draws = tf.reshape(tf.cast(tf.linspace(-2.0, 2.0, 800), tf.float64), [100, 4, 2])
    result = comparison.direct_posterior_agreement(
        draws,
        draws,
        parameter_names=("a", "b"),
        learned_diagnostic=_diagnostic((300.0, 280.0)),
        exact_diagnostic=_diagnostic((300.0, 280.0)),
    )
    assert result["passed"] is True
    assert result["status"] == "PASS"
    assert all(row["intervals_overlap"] for row in result["rows"])


def test_direct_posterior_agreement_fails_large_shift():
    exact = tf.reshape(tf.cast(tf.linspace(-2.0, 2.0, 800), tf.float64), [100, 4, 2])
    learned = exact + tf.constant([2.0, 0.0], tf.float64)
    result = comparison.direct_posterior_agreement(
        learned,
        exact,
        parameter_names=("a", "b"),
        learned_diagnostic=_diagnostic((300.0, 280.0)),
        exact_diagnostic=_diagnostic((300.0, 280.0)),
    )
    assert result["passed"] is False
    assert result["status"] == "FAIL"
    assert result["rows"][0]["status"] == "FAIL"


def test_cost_ledger_charges_supervision_and_training_and_computes_break_even():
    supervision = {"supervision_generation_seconds": 5.0}
    training = {
        "cost": {
            "grid_wall_seconds": 7.0,
            "screen_optimization_seconds": 3.0,
            "final_optimization_seconds": 2.0,
        }
    }
    tuning = {
        "learned_residual": {"rows": ({"elapsed_seconds": 4.0},)},
        "true_gradient": {"rows": ({"elapsed_seconds": 6.0},)},
    }
    diagnostic = {
        "min_bulk_ess": 100.0,
        "parameter_diagnostics": ({"parameter": "x", "bulk_ess": 100.0},),
    }
    runs = {
        "learned_residual": {
            "sampling_execution_seconds": 10.0,
            "retained_checks": ({"full_convergence": diagnostic},),
        },
        "true_gradient": {
            "sampling_execution_seconds": 40.0,
            "retained_checks": ({"full_convergence": diagnostic},),
        },
    }
    matched = {
        "warm_median_seconds": {"learned_residual": 2.0, "true_gradient": 12.0},
        "mechanics": {"transitions_per_chain": 5},
    }
    result = comparison.cost_ledger(
        supervision=supervision,
        training=training,
        tuning=tuning,
        runs=runs,
        matched=matched,
    )
    assert result["reuse_scenario_seconds"]["learned_residual"] == 26.0
    assert result["reuse_scenario_seconds"]["true_gradient"] == 46.0
    assert result["hnn_preparation_seconds"] == 12.0
    assert result["preparation_break_even_transition_batches"] == 6
    assert result["reuse_campaign_pre_sampling_delta_seconds"] == 10.0
    assert result["reuse_campaign_break_even_transition_batches"] == 5
    assert result["break_even_transition_batches"] == 5
    assert result["from_scratch_total_seconds"] is None


def test_cost_ledger_separates_immediate_campaign_break_even_from_hnn_preparation():
    supervision = {"supervision_generation_seconds": 5.0}
    training = {
        "cost": {
            "grid_wall_seconds": 7.0,
            "screen_optimization_seconds": 3.0,
            "final_optimization_seconds": 2.0,
        }
    }
    tuning = {
        "learned_residual": {"rows": ({"elapsed_seconds": 4.0},)},
        "true_gradient": {"rows": ({"elapsed_seconds": 30.0},)},
    }
    diagnostic = {
        "min_bulk_ess": 100.0,
        "parameter_diagnostics": ({"parameter": "x", "bulk_ess": 100.0},),
    }
    runs = {
        name: {
            "sampling_execution_seconds": 10.0,
            "retained_checks": ({"full_convergence": diagnostic},),
        }
        for name in comparison.PRIMARY_ARMS
    }
    matched = {
        "warm_median_seconds": {"learned_residual": 2.0, "true_gradient": 12.0},
        "mechanics": {"transitions_per_chain": 5},
    }

    result = comparison.cost_ledger(
        supervision=supervision,
        training=training,
        tuning=tuning,
        runs=runs,
        matched=matched,
    )

    assert result["hnn_preparation_seconds"] == 12.0
    assert result["preparation_break_even_transition_batches"] == 6
    assert result["reuse_campaign_pre_sampling_delta_seconds"] == -14.0
    assert result["reuse_campaign_break_even_transition_batches"] == 0
    assert result["break_even_transition_batches"] == 0


def test_cost_ledger_preserves_failed_arm_without_inventing_efficiency():
    supervision = {"supervision_generation_seconds": 1.0}
    training = {
        "cost": {
            "grid_wall_seconds": 2.0,
            "screen_optimization_seconds": 1.0,
            "final_optimization_seconds": 1.0,
        }
    }
    tuning = {
        name: {"rows": ({"elapsed_seconds": 1.0},)}
        for name in comparison.PRIMARY_ARMS
    }
    diagnostic = {
        "min_bulk_ess": 100.0,
        "parameter_diagnostics": ({"parameter": "x", "bulk_ess": 100.0},),
    }
    runs = {
        "learned_residual": {
            "sampling_execution_seconds": 3.0,
            "retained_checks": (),
        },
        "true_gradient": {
            "sampling_execution_seconds": 4.0,
            "retained_checks": ({"full_convergence": diagnostic},),
        },
    }
    matched = {
        "warm_median_seconds": {"learned_residual": 2.0, "true_gradient": 3.0},
        "mechanics": {"transitions_per_chain": 5},
    }
    result = comparison.cost_ledger(
        supervision=supervision,
        training=training,
        tuning=tuning,
        runs=runs,
        matched=matched,
    )
    assert result["minimum_bulk_ess"]["learned_residual"] is None
    assert result["tuned_seconds_per_minimum_bulk_ess"]["learned_residual"] is None
    assert result["efficiency_status"]["learned_residual"] == "not_valid_for_efficiency"
    assert comparison._strictly_lower_finite(None, 1.0) is False


def test_timing_paths_require_explicit_synchronization_and_alternating_repeats():
    matched = inspect.getsource(comparison.matched_mechanics_benchmark)
    training = inspect.getsource(
        __import__(
            "bayesfilter.inference.neural_force_training", fromlist=["train_scalar_residual_force"]
        ).train_scalar_residual_force
    )
    assert "synchronize_chain" in matched
    assert "reversed(PRIMARY_ARMS)" in matched
    assert ".numpy()" in training


def test_runner_configures_gpu_memory_before_importing_comparison_module():
    from pathlib import Path

    source = Path(
        "docs/benchmarks/run_hnn_neutra_exact_comparison_2026_07_18.py"
    ).read_text(encoding="utf-8")
    configure = source.index("memory = configure_tensorflow_gpu_memory_growth")
    comparison_import = source.index(
        "from bayesfilter.testing import hnn_neutra_exact_comparison_tf"
    )
    assert configure < comparison_import


def test_reviewed_tuning_grids_are_explicit_for_all_four_cells():
    pp = {"cell": "PP-UKF"}
    sir = {
        "cell": "SIR-SGQF",
        "step_sizes": (0.2, 0.4, 0.6, 0.8),
        "leapfrog_steps": (6, 10),
    }
    structural = {
        "cell": "STR-UKF",
        "step_sizes": (0.025, 0.05, 0.1, 0.2),
        "leapfrog_steps": (8, 12),
    }
    assert comparison._tuning_grid(pp) == ((0.2, 0.4, 0.6, 0.8), (6, 10))
    assert comparison._tuning_grid(sir) == ((0.2, 0.4, 0.6, 0.8), (6, 10))
    assert comparison._tuning_grid(structural) == ((0.025, 0.05, 0.1, 0.2), (8, 12))


def test_comparison_uses_native_tuner_and_never_calls_legacy_selector():
    full_source = inspect.getsource(comparison.run_full_cell)
    arm_source = inspect.getsource(comparison._native_tune_arm)
    assert "_native_tune_arm" in full_source
    assert "tune_fixed_transport_hmc_kernel" in arm_source
    assert "campaign.tune_force" not in full_source
    assert "campaign.tune_force" not in arm_source
    assert "target_accept_prob=0.70" in arm_source
    assert "acceptance_band=(0.65, 0.75)" in arm_source
    assert "budget_schedule=(16, 32, 64, 128, 256)" in arm_source
    assert "require_modern_rank_normalized_verification=True" in arm_source
    assert "verification_min_retained_results_per_chain=1000" in arm_source


def test_canary_runs_both_arms_through_native_tuning(monkeypatch, tmp_path: Path):
    calls = []

    class _Binding:
        def hmc_target(self):
            return object()

    class _Result:
        def payload(self):
            return {"identity_z_mass_artifact_signature": "same-mass"}

    def fake_native_tune_arm(**kwargs):
        calls.append(kwargs)
        return {
            "passed": True,
            "result": _Result(),
            "selected_payload": {
                "step_size": 0.1,
                "num_leapfrog_steps": 6,
                "acceptance_rate": 0.70,
            },
        }

    monkeypatch.setattr(
        comparison,
        "validate_value_only_endpoint_parity",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        comparison,
        "force_arms",
        lambda *_args: {
            "learned_residual": SimpleNamespace(identity="hnn"),
            "true_gradient": SimpleNamespace(identity="exact"),
        },
    )
    monkeypatch.setattr(comparison, "_native_tune_arm", fake_native_tune_arm)
    context = {
        "cell": "PP-UKF",
        "latent": tf.zeros((64, comparison.predator_prey.DIMENSION), tf.float64),
        "binding": _Binding(),
        "loaded": SimpleNamespace(
            manifest=SimpleNamespace(transport_hash="transport-hash")
        ),
    }
    monkeypatch.setattr(comparison, "_target_signature", lambda _context: "target")

    result = comparison.run_canary(context, object(), tmp_path)

    assert result["passed"] is True
    assert result["schema"] == "bayesfilter.hnn_neutra_native_tuning_canary.v2"
    assert tuple(result["arms"]) == comparison.PRIMARY_ARMS
    assert len(calls) == 2
    assert all(call["raise_on_failure"] is False for call in calls)
    assert calls[0]["output_root"] != calls[1]["output_root"]
    assert result["tuning_contract"]["target_accept_prob"] == 0.70
    assert result["tuning_contract"]["acceptance_band"] == (0.65, 0.75)
    assert result["cross_arm_identity_z_mass_signature_match"] is True
