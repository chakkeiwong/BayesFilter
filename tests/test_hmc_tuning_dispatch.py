"""Focused contracts for the public HMC tuning dispatcher."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import tensorflow as tf

from bayesfilter.inference import (
    BoundRetainedHMCArchiveConfig,
    DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
    FourChainMeanBandAcceptancePolicy,
    FrozenPositionOnlyForce,
    FrozenTargetPotential,
    TensorFlowHMCKernelTuningConfig,
    bind_neural_force_hmc_tuning_runner,
    build_retained_bound_hmc_archive_runner_from_tuning_result,
    load_tensorflow_hmc_tuning_result,
    tune_hmc_kernel,
)
from bayesfilter.inference import hmc_kernel_tuning
from bayesfilter.inference import hmc_tensorflow_tuning
from bayesfilter.inference import hmc_tuning_dispatch


class _Adapter:
    target_scope = "tensorflow-dispatch-test"

    def adapter_signature(self) -> str:
        return "tensorflow-dispatch-test-adapter-v1"


def _explicit_diagnostic_numerics() -> dict[str, Any]:
    return {
        "target_accept_prob": 0.70,
        "verification_repair_rounds": 0,
        "step_repair_factor": 2.0,
        "mass_shrinkage": 0.10,
        "covariance_jitter": 1.0e-9,
        "eigenvalue_floor": 1.0e-9,
        "max_condition_number": 1.0e8,
        "seed": (20260828, 1),
    }


def _declared_band_policy() -> FourChainMeanBandAcceptancePolicy:
    return FourChainMeanBandAcceptancePolicy(
        overall_band=(0.65, 0.75),
        per_chain_band=(0.55, 0.85),
    )


def _binding():
    return bind_neural_force_hmc_tuning_runner(
        force=FrozenPositionOnlyForce(
            lambda position: position,
            identity="tensorflow-dispatch-test-force-v1",
            semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
        ),
        target=FrozenTargetPotential(
            lambda position: 0.5
            * tf.reduce_sum(tf.square(position), axis=-1),
            identity="tensorflow-dispatch-test-target-v1",
        ),
        target_scope="tensorflow-dispatch-test",
    )


def _config() -> TensorFlowHMCKernelTuningConfig:
    return TensorFlowHMCKernelTuningConfig(
        parameter_dimension=2,
        evidence_role="diagnostic_only",
        mass_window_results=(1,),
        step_adaptation_results=1,
        verification_results=1,
        max_leapfrog_steps=1,
        initial_step_size=0.1,
        budget_provenance="one-step interface diagnostic",
        initial_step_size_provenance="convenience diagnostic",
        geometry_provenance="unit parameter scales diagnostic",
        target_scope="tensorflow-dispatch-test",
        acceptance_policy=FourChainMeanBandAcceptancePolicy(
            overall_band=(0.0, 1.0),
            per_chain_band=(0.0, 1.0),
        ),
        **_explicit_diagnostic_numerics(),
    )


def test_legacy_dispatch_calls_private_implementation_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    observed: dict[str, Any] = {}

    def fake_implementation(**kwargs: Any) -> object:
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        hmc_kernel_tuning, "_run_canonical_hmc_tuning", fake_implementation
    )
    result = hmc_tuning_dispatch.tune_hmc_kernel(
        adapter="adapter",
        initial_position=(0.0,),
    )

    assert result is sentinel
    assert observed["adapter"] == "adapter"


def test_tensorflow_config_limits_authority_to_mechanics_handoff() -> None:
    payload = _config().payload()

    assert payload["artifact_authority"] is False
    assert payload["posterior_admission_authority"] is False
    assert payload["admission_supported"] is False
    assert payload["mechanics_handoff_supported"] is True
    assert payload["chain_count"] == 4
    assert payload["dtype"] == "float64"
    assert payload["initial_chain_state_policy"] == (
        "four_identical_zero_states_in_current_affine_coordinates"
    )
    assert payload["mass_window_leapfrog_steps"] == 1
    assert payload["trajectory_candidate_policy"] == (
        "powers_of_two_then_explicit_cap"
    )
    assert payload["dual_averaging_internal_policy"].startswith(
        "tensorflow_probability_defaults_except"
    )
    assert payload["handoff_eligibility"] == "result_dependent_candidate_screen"
    assert payload["fresh_rhat_verification"] == "not_part_of_tuning_handoff"
    assert payload["rhat_role"] == "retained_explanatory_only"
    assert payload["xla_qualification"] == "not_required_for_non_xla_execution"
    assert payload["xla_mode"] == "disabled_unless_separately_qualified"
    dispatch_source = inspect.getsource(hmc_tuning_dispatch)
    numerical_source = inspect.getsource(hmc_tensorflow_tuning)
    assert "import numpy" not in dispatch_source
    assert "np." not in dispatch_source
    assert "import numpy" not in numerical_source
    assert "np." not in numerical_source
    assert len(dispatch_source.splitlines()) < 150
    assert hmc_tensorflow_tuning.TensorFlowHMCKernelTuningConfig is (
        hmc_tuning_dispatch.TensorFlowHMCKernelTuningConfig
    )


def test_tensorflow_diagnostic_graph_smoke_cannot_issue_handoff() -> None:
    binding = _binding()
    result = tune_hmc_kernel(
        adapter=_Adapter(),
        initial_position=tf.zeros([2], tf.float64),
        parameter_scales=tf.ones([2], tf.float64),
        config=_config(),
        runner_binding=binding,
    )

    assert bool(result.health_passed) is True
    assert bool(result.handoff_eligible) is False
    assert bool(result.passed) is False
    assert result.final_raw_state.shape == (4, 2)
    graph_def = result.graph_function.get_concrete_function().graph.as_graph_def()
    operations = {node.op for node in graph_def.node}
    operations.update(
        node.op for function in graph_def.library.function for node in function.node_def
    )
    assert operations.isdisjoint(
        {"PyFunc", "PyFuncStateless", "EagerPyFunc", "ParallelFor"}
    )
    with pytest.raises(ValueError, match="candidate mechanics screen"):
        build_retained_bound_hmc_archive_runner_from_tuning_result(
            tuning_result=result,
            runner_binding=binding,
        )


def _log_probabilities(values: list[float]) -> tf.Tensor:
    tensor = tf.constant([values], tf.float64)
    return tf.math.log(tensor)


@pytest.mark.parametrize(
    "probabilities",
    (
        [0.55, 0.65, 0.70, 0.70],
        [0.85, 0.75, 0.70, 0.70],
    ),
)
def test_four_chain_policy_includes_both_boundaries(
    probabilities: list[float],
) -> None:
    decision = _declared_band_policy().evaluate(
        _log_probabilities(probabilities), divergence_count=0
    )

    assert bool(decision.passed) is True


def test_four_chain_policy_rejects_two_ulps_outside_inclusive_boundary() -> None:
    one_below = tf.math.nextafter(
        tf.constant(0.55, tf.float64), tf.constant(0.0, tf.float64)
    )
    below = tf.math.nextafter(one_below, tf.constant(0.0, tf.float64))
    probabilities = tf.stack(
        (
            below,
            tf.constant(0.75, tf.float64),
            tf.constant(0.75, tf.float64),
            tf.constant(0.75, tf.float64),
        )
    )[tf.newaxis, :]
    decision = _declared_band_policy().evaluate(
        tf.math.log(probabilities), divergence_count=0
    )

    assert bool(decision.overall_band_pass) is True
    assert bool(decision.per_chain_band_passes[0]) is False
    assert bool(decision.passed) is False


@pytest.mark.parametrize(
    ("probabilities", "direction"),
    (
        ([0.90, 0.60, 0.60, 0.60], 1),
        ([0.50, 0.70, 0.70, 0.70], -1),
        ([0.50, 0.90, 0.70, 0.70], 0),
    ),
)
def test_four_chain_policy_rejects_a_hidden_bad_chain(
    probabilities: list[float], direction: int
) -> None:
    decision = _declared_band_policy().evaluate(
        _log_probabilities(probabilities), divergence_count=0
    )

    assert bool(decision.overall_band_pass) is True
    assert bool(decision.passed) is False
    assert int(decision.repair_direction) == direction


def test_four_chain_policy_handles_infinities_and_divergence_exactly() -> None:
    policy = _declared_band_policy()
    base = tf.fill([1, 4], tf.math.log(tf.constant(0.70, tf.float64)))

    support_rejection = tf.tensor_scatter_nd_update(
        base, indices=[[0, 0]], updates=[tf.constant(float("-inf"), tf.float64)]
    )
    negative_infinity = policy.evaluate(support_rejection, divergence_count=0)
    assert bool(negative_infinity.log_acceptance_defined) is True
    assert float(negative_infinity.chain_means[0]) == 0.0

    for invalid in (float("nan"), float("inf")):
        values = tf.tensor_scatter_nd_update(
            base, indices=[[0, 0]], updates=[tf.constant(invalid, tf.float64)]
        )
        decision = policy.evaluate(values, divergence_count=0)
        assert bool(decision.log_acceptance_defined) is False
        assert bool(decision.passed) is False

    divergent = policy.evaluate(base, divergence_count=1)
    assert bool(divergent.divergence_pass) is False
    assert bool(divergent.passed) is False


def test_candidate_dense_metric_requires_enough_states() -> None:
    with pytest.raises(ValueError, match=r"at least d \+ 1 states"):
        TensorFlowHMCKernelTuningConfig(
            parameter_dimension=54,
            evidence_role="candidate",
            mass_window_results=(1,),
            step_adaptation_results=1,
            verification_results=1,
            max_leapfrog_steps=1,
            initial_step_size=0.1,
            budget_provenance="rank contract fixture",
            initial_step_size_provenance="fixture",
            geometry_provenance="fixture",
            target_scope="tensorflow-dispatch-test",
            acceptance_policy=_declared_band_policy(),
            **_explicit_diagnostic_numerics(),
        )


def test_failed_search_reports_last_real_verification_not_synthetic_health() -> None:
    binding = bind_neural_force_hmc_tuning_runner(
        force=FrozenPositionOnlyForce(
            lambda position: tf.zeros_like(position),
            identity="flat-force-v1",
            semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
        ),
        target=FrozenTargetPotential(
            lambda position: tf.zeros(tf.shape(position)[:-1], tf.float64),
            identity="flat-target-v1",
        ),
        target_scope="tensorflow-dispatch-test",
    )
    config = TensorFlowHMCKernelTuningConfig(
        parameter_dimension=2,
        evidence_role="diagnostic_only",
        mass_window_results=(1,),
        step_adaptation_results=1,
        verification_results=1,
        max_leapfrog_steps=1,
        initial_step_size=0.01,
        budget_provenance="failed-selection telemetry fixture",
        initial_step_size_provenance="fixture",
        geometry_provenance="fixture",
        target_scope="tensorflow-dispatch-test",
        acceptance_policy=FourChainMeanBandAcceptancePolicy(
            overall_band=(0.0, 0.0), per_chain_band=(0.0, 0.0)
        ),
        **_explicit_diagnostic_numerics(),
    )

    result = tune_hmc_kernel(
        adapter=_Adapter(),
        initial_position=tf.zeros([2], tf.float64),
        parameter_scales=tf.ones([2], tf.float64),
        config=config,
        runner_binding=binding,
    )

    assert int(result.selected_candidate_index) == -1
    assert int(result.reported_candidate_index) == 0
    assert bool(result.candidate_selected) is False
    tf.debugging.assert_equal(
        result.acceptance_decision.chain_means, tf.ones([4], tf.float64)
    )


def test_candidate_artifact_reloads_and_runs_bound_retained_continuation(
    tmp_path: Path,
) -> None:
    binding = _binding()
    config = TensorFlowHMCKernelTuningConfig(
        parameter_dimension=2,
        evidence_role="candidate",
        mass_window_results=(1,),
        step_adaptation_results=1,
        verification_results=2,
        max_leapfrog_steps=1,
        initial_step_size=0.01,
        budget_provenance="bounded candidate contract fixture",
        initial_step_size_provenance="small Gaussian fixture",
        geometry_provenance="unit Gaussian scales",
        target_scope="tensorflow-dispatch-test",
        acceptance_policy=FourChainMeanBandAcceptancePolicy(
            overall_band=(0.0, 1.0), per_chain_band=(0.0, 1.0)
        ),
        **_explicit_diagnostic_numerics(),
    )
    result = tune_hmc_kernel(
        adapter=_Adapter(),
        initial_position=tf.constant([0.2, -0.3], tf.float64),
        parameter_scales=tf.ones([2], tf.float64),
        config=config,
        runner_binding=binding,
        output_dir=tmp_path / "tuning",
    )

    assert bool(result.candidate_selected) is True
    assert bool(result.metric_update_valid) is True
    assert int(result.metric_update_count) == 1
    assert bool(result.heuristic_screen_passed) is True
    assert result.posterior_admission_authority is False
    assert result.admission_supported is False
    assert result.mechanics_handoff_supported is True
    assert bool(result.handoff_eligible) is True
    assert bool(result.passed) is True

    loaded = load_tensorflow_hmc_tuning_result(
        result.artifact_manifest_path,
        adapter=_Adapter(),
        runner_binding=binding,
    )
    assert bool(loaded.heuristic_screen_passed) is True
    assert loaded.posterior_admission_authority is False
    assert loaded.admission_supported is False
    assert loaded.mechanics_handoff_supported is True
    assert bool(loaded.handoff_eligible) is True
    assert bool(loaded.passed) is True

    runner = build_retained_bound_hmc_archive_runner_from_tuning_result(
        tuning_result=loaded,
        runner_binding=binding,
    )
    pilot = runner.run(
        BoundRetainedHMCArchiveConfig(
            num_results=2,
            seed=(20260828, 2),
            output_dir=tmp_path / "pilot",
            budget_provenance="two-draw retained mechanics fixture",
        )
    )
    tf.debugging.assert_equal(pilot.initial_chain_state, loaded.final_chain_state)
    assert pilot.binding_hash == binding.binding_hash

    extension = runner.run(
        BoundRetainedHMCArchiveConfig(
            num_results=2,
            seed=(20260828, 3),
            output_dir=tmp_path / "extension",
            budget_provenance="two-draw continuation mechanics fixture",
            continuation_manifest=pilot.archive_manifest_path,
        )
    )
    tf.debugging.assert_equal(
        extension.initial_chain_state, pilot.final_chain_state
    )
