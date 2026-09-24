from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.highdim.ledh_pfpf_genut_initial_rqmc_tf import (
    finite_value_standard_score_initial_rqmc,
)
from docs.benchmarks import run_ledh_pfpf_genut_full_sqmc_full_horizons as runner
from docs.benchmarks.run_ledh_pfpf_genut_full_sqmc_all_models import (
    campaign_inputs as prefix_inputs,
    state_map as prefix_state_map,
)
from docs.benchmarks.run_ledh_pfpf_genut_initial_rqmc_all_models import (
    build_campaign_models as build_prefix_models,
)


def test_full_horizon_inventory_and_event_orders() -> None:
    models = runner.build_full_horizon_models(include_references=False)
    assert [model.row_id for model in models] == list(runner.MODEL_HORIZONS)
    expected_transition_first = {
        "lgssm_T50": False,
        "ksc_sv_T10": False,
        "exact_sv_T10": False,
        "generalized_sv_T10": True,
        "predator_prey_T20": True,
        "austria_sir_T20": True,
    }
    for model in models:
        horizon = runner.MODEL_HORIZONS[model.row_id]
        assert model.observations.shape == (
            horizon,
            model.callbacks.observation_dimension,
        )
        assert (
            model.callbacks.transition_before_first_observation
            is expected_transition_first[model.row_id]
        )


def test_full_horizon_inputs_pair_the_intended_arms() -> None:
    for model in runner.build_full_horizon_models(include_references=False):
        inputs = runner.campaign_inputs(model, 95123)
        horizon = runner.MODEL_HORIZONS[model.row_id]
        process_steps = (
            horizon
            if model.callbacks.transition_before_first_observation
            else horizon - 1
        )
        dimension = model.callbacks.state_dimension
        for arm in runner.ARMS:
            assert inputs[arm]["process"].shape == (
                process_steps,
                runner.PARTICLE_COUNT,
                dimension,
            )
            assert inputs[arm]["ancestors"].shape == (
                process_steps,
                runner.PARTICLE_COUNT,
            )
        assert (
            inputs["iid_existing"]["process_sha256"]
            == inputs["initial_rqmc"]["process_sha256"]
        )
        assert (
            inputs["initial_rqmc"]["initial_sha256"]
            == inputs["full_sqmc_halton"]["initial_sha256"]
        )
        assert len(inputs["full_sqmc_halton"]["raw_joint_sha256"]) == process_steps


def test_chapter18b_ledger_is_fail_closed_and_does_not_add_state_noise() -> None:
    ledger = runner._chapter18b_ledger()  # noqa: SLF001
    assert ledger["full_horizon"] == 100
    assert ledger["historical_particle_count"] == 1002
    assert ledger["historical_test_status"] == "previously_tested_T100_failed_candidate"
    assert ledger["full_sqmc_standard_score_status"] == "not_implemented_not_executed"
    assert ledger["artificial_state_noise_added"] is False
    assert ledger["score_substitution_used"] is False


def test_runner_uses_standard_score_without_autodiff_or_finite_difference() -> None:
    source = inspect.getsource(runner)
    assert "finite_value_standard_score_initial_rqmc" in source
    assert "repository_standard_pairwise_backward_filtering_score" in source
    for forbidden in ("GradientTape", "ForwardAccumulator"):
        assert forbidden not in source


def test_diagnostic_policy_controls_are_explicit_and_default_preserving() -> None:
    import inspect as _inspect
    from bayesfilter.highdim import ledh_pfpf_genut_initial_rqmc_tf as core

    signature = _inspect.signature(core.finite_value_standard_score_initial_rqmc)
    assert signature.parameters["state_map_policy"].default == "adaptive_empirical"
    assert signature.parameters["reset_policy"].default == "contract_e"
    assert signature.parameters["ancestor_uniform_policy"].default == "supplied"
    assert core.RESET_POLICIES == ("contract_e", "ot_only", "none")
    assert core.STATE_MAP_POLICIES == ("adaptive_empirical", "fixed_supplied")


def test_functional_loop_matches_unrolled_t6_for_both_ancestry_policies() -> None:
    model = build_prefix_models(include_references=False)[0]
    inputs_by_arm = prefix_inputs(model, 95100)
    location, scale = prefix_state_map(model)
    design = runner._design(model.callbacks.state_dimension)  # noqa: SLF001
    for policy, arm in (
        ("existing_one_to_one", "initial_rqmc"),
        ("hilbert_inverse_cdf", "full_sqmc_halton"),
    ):
        inputs = inputs_by_arm[arm]

        def evaluate(functional: bool):
            return finite_value_standard_score_initial_rqmc(
                model.callbacks,
                model.theta,
                model.observations,
                inputs["initial"],
                inputs["process"],
                design,
                ancestry_policy=policy,
                process_ancestor_uniforms=inputs["ancestors"],
                state_map_location=location,
                state_map_scale=scale,
                functional_time_loop=functional,
                epsilon=runner.EPSILON,
                sinkhorn_steps=runner.SINKHORN_STEPS,
                balance_steps=runner.BALANCE_STEPS,
                ridge=runner.RIDGE,
            )

        unrolled = evaluate(False)
        functional = evaluate(True)
        tf.debugging.assert_equal(unrolled[0], functional[0])
        tf.debugging.assert_equal(unrolled[1], functional[1])
        assert tuple(unrolled[2]) == tuple(functional[2])
        for key in unrolled[2]:
            tf.debugging.assert_equal(unrolled[2][key], functional[2][key])


def test_exact_sv_full_horizon_full_sqmc_cpu_xla_smoke() -> None:
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("test requires CUDA_VISIBLE_DEVICES=-1 before import")
    model = runner.build_full_horizon_models(include_references=False)[2]
    evaluator = runner.make_evaluator(model, full_sqmc=True)
    inputs = runner.campaign_inputs(model, 95100)["full_sqmc_halton"]
    location, scale = runner.state_map(model)
    value, score, diagnostics = evaluator(
        model.theta,
        model.observations,
        inputs["initial"],
        inputs["process"],
        inputs["ancestors"],
        location,
        scale,
        runner._design(model.callbacks.state_dimension),  # noqa: SLF001
    )
    concrete = evaluator.get_concrete_function()
    must_compile = concrete.function_def.attr.get("_XlaMustCompile")
    assert must_compile is not None and must_compile.b
    assert evaluator.experimental_get_tracing_count() == 1
    assert "CPU:0" in value.device
    assert bool(diagnostics["program_valid"].numpy())
    tf.debugging.assert_all_finite(value, "value")
    tf.debugging.assert_all_finite(score, "score")
