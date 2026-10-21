"""Actual public pipeline wiring; CPU mechanics only, no posterior claims."""
from dataclasses import replace

import pytest

from bayesfilter.testing.inference_validation.mams_adaptation_consumer import run_adapted_mams_search
from tests.test_hmc_mams_trajectory_proposals import config_for
from tests.test_modern_hmc_pipeline import setup


@pytest.mark.parametrize("transport", [False, True])
def test_adapted_consumer_uses_public_authority_and_reloads_all_members(tmp_path, transport):
    binding, template = setup(family="mams", transport=transport)
    pilot = config_for(binding, max_candidates=2, duration_factors=(1., 2.))
    search = replace(template, repair_trajectory_policy="preserve_mean_duration",
                     repair_mean_duration_max_steps=pilot.max_steps_cap)
    proposals, run, members = run_adapted_mams_search(binding, pilot, search,
                                                    output_dir=tmp_path/"search")
    assert len(run.result.candidates) == len(proposals["candidates"])
    assert len(members) == 2
    assert set(members) == set(run.result.verified_candidate_ids)
    assert run.result.payload()["nominee_id"] is None
    from bayesfilter.inference import run_hmc_posterior, SequentialNeuTraHMCConfig
    from bayesfilter.runtime.execution_budget import execution_budget, ExecutionBudgetExceeded
    first_member = next(iter(members.values()))
    config = SequentialNeuTraHMCConfig(step_size=first_member.step_size,
        num_leapfrog_steps=first_member.num_leapfrog_steps, jit_compile=False,
        warmup_seed=(41, 51), retained_seed=(41, 52))
    with execution_budget(check=lambda: False):
        with pytest.raises(ExecutionBudgetExceeded):
            run_hmc_posterior(member=first_member, config=config, parameter_names=("x", "y"))
    for index, member in enumerate(members.values()):
        continuation = member.run(num_results=4, seed=(20261011, 831+index),
                                  output_dir=tmp_path/f"retained-{index}")
        assert tuple(continuation["samples"].shape) == (4, 4, 2)


@pytest.mark.parametrize("target", ["LGSSM", "NL-ACC"])
def test_state_space_mams_proposer_public_tuner_and_replay(target, tmp_path):
    import tensorflow as tf
    from bayesfilter.inference import PrecomputedMassArtifact, bind_hmc_candidate_set_execution
    from bayesfilter.inference.posterior_adapter import value_score_capability
    from scripts.run_modern_hmc_bridge import _fixture
    fixture, starts, sources, metadata = _fixture(target, use_xla=False)
    base, template = setup(family="mams")
    dimension = int(starts.shape[-1])
    identity = tf.eye(dimension, dtype=tf.float64)
    mass = PrecomputedMassArtifact(position=tf.zeros(dimension, tf.float64),
        covariance=identity, factor=identity, adapter_signature=fixture.adapter.adapter_signature(),
        position_role="reference_center", covariance_source="explicit pipeline mechanics identity")
    execution = replace(base.config, target_status_trace_policy=("per_chain_step"
        if callable(getattr(fixture.adapter, "target_status_telemetry", None)) else "none"))
    binding = bind_hmc_candidate_set_execution(adapter=fixture.adapter, initial_position=starts,
        target_scope=value_score_capability(fixture.adapter).target_scope,
        target_lineage=dict(model=target, data=metadata["data_sha256"]),
        config=execution, source_paths=[__file__, *sources],
        scope_id="mams-adaptation-"+target, search_id="cpu-mechanics",
        epsilon_domain=(.001, 2.), repair_factor=1.25, max_repairs_per_family=1,
        mass_artifact=mass)
    # Short frozen windows and broad acceptance floor isolate real filter/score
    # wiring, not mixing. Model-specific values are explicit fixture inputs.
    pilot = config_for(binding, initial_epsilon=.03, initial_mean_duration=.09,
        max_requested_duration=.18, max_candidates=1, duration_factors=(1.,),
        epsilon_adaptation_windows=1, fixed_window_transitions=16,
        frozen_coordinate_scale=(1.,)*dimension, frozen_projection_directions=(),
        max_pilot_wall_seconds=180.)
    search = replace(template, primary_l_grid=(5,), epsilon_by_l=((5, (.03,)),),
                     repair_trajectory_policy="preserve_mean_duration",
                     repair_mean_duration_max_steps=pilot.max_steps_cap)
    proposals, run, members = run_adapted_mams_search(binding, pilot, search,
                                                    output_dir=tmp_path/target)
    assert proposals["candidates"]
    assert run.result.candidates
    assert set(members) == set(run.result.verified_candidate_ids)
    assert members, run.result.candidate_states
    for index, member in enumerate(members.values()):
        continuation = member.run(num_results=4, seed=(20261011, 1201+index),
                                  output_dir=tmp_path/f"retained-{index}")
        assert tuple(continuation["samples"].shape) == (4, 4, dimension)


def test_consumer_invalid_cap_fails_before_creating_output(tmp_path):
    binding, template = setup(family="mams")
    pilot = config_for(binding, max_candidates=3)
    search = replace(template, repair_trajectory_policy="preserve_mean_duration",
                     repair_mean_duration_max_steps=pilot.max_steps_cap)
    with pytest.raises(ValueError, match="candidate cap"):
        run_adapted_mams_search(binding, pilot, search, output_dir=tmp_path/"invalid")
    assert not (tmp_path/"invalid").exists()
