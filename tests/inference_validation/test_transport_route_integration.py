"""Public route composition on several laws; small CPU mechanics fixtures only."""
from dataclasses import replace
import json

import pytest
import tensorflow as tf

from bayesfilter.testing.inference_validation.designs import ScenarioSpec, ValidationDesign
from bayesfilter.testing.inference_validation.procedures import execute_pipeline, initial_starts
from bayesfilter.testing.inference_validation.targets import ValidationTarget
from bayesfilter.testing.inference_validation.storage import read_json, read_tensor, file_hash
from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory


@pytest.mark.parametrize("target_id", ["gaussian", "banana", "dirichlet"])
def test_nonlinear_frozen_transport_retains_reloads_and_excludes_warmup(tmp_path, target_id):
    from tests.test_dense_iaf_neutra_artifact_loader import _payload

    target = ValidationTarget(target_id, jit_compile=False)
    payload = _payload(target_signature=target.adapter_signature())
    design = ValidationDesign(design_id="nonlinear-route-" + target_id, engine="accuracy",
        scenario=ScenarioSpec(target_id, "fixed_transport"), replications=1, draws=64,
        seed=2026092197, budget_seconds=150, device="cpu_reference",
        purpose="nonlinear public route mechanics, no transport quality claim",
        numerical_provenance="M17 engineering plan; inherited fixed-weight fixture",
        l_grid=(2, 3), step_size=.8, posterior_cap=128,
        options={"transport_payload": payload,
            "acceptance_policy": {"practical_region": (.41, .99), "repair_region": (.405, .995)},
            "search": {"pilot_enabled": False, "refinement_rounds": 0,
                       "total_budget_units": 24, "repair_reserve_units": 4, "evidence_rungs": (1,)}})
    output = execute_pipeline(design, tmp_path)
    inventory = check_inventory(read_json(output["tuning_path"]))
    assert inventory["finding"] == "inventory_passed"
    assert output["verified_candidate_ids"]
    assert {m["candidate_id"] for m in output["members"]} == set(output["verified_candidate_ids"])
    for member in output["members"]:
        assert member["status"] == "assessed"
        assert member["warmup_exclusion_matches"]
        draws = read_tensor(member["draws_path"])
        assert draws.shape[-1] == len(target.spec.parameters)
        assert draws.shape[0] == member["posterior"]["retained_results_per_chain"]
        if target_id == "dirichlet" and draws.shape[0]:
            tf.debugging.assert_near(tf.reduce_sum(draws, -1), tf.ones(draws.shape[:2], tf.float64))
    hashes = {m["candidate_id"]: file_hash(m["draws_path"]) for m in output["members"]}
    resumed = execute_pipeline(design, tmp_path)
    assert hashes == {m["candidate_id"]: file_hash(m["draws_path"]) for m in resumed["members"]}


@pytest.mark.parametrize("target_id,data", [("gaussian", None), ("beta_binomial", [7, 12]),
                                           ("lgssm_location", [1., -1., .5]), ("banana", None)])
def test_position_field_multimodel_resume_remains_conditional(tmp_path, target_id, data):
    from bayesfilter.inference import (FourChainMeanBandAcceptancePolicy, FrozenPositionOnlyForce,
        FrozenTargetPotential, DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
        bind_neural_force_hmc_tuning_runner, tune_hmc_kernel, resume_position_field_candidate_tuning,
        HMCControllerConfig, build_retained_bound_hmc_archive_runner_from_candidate_set_result)
    from tests.test_hmc_tuning_dispatch import _config

    target = ValidationTarget(target_id, data=data, jit_compile=False)
    binding = bind_neural_force_hmc_tuning_runner(
        force=FrozenPositionOnlyForce(lambda q: -.8 * target.log_prob_and_grad(q)[1],
            identity="m17-nonexact-force-" + target.adapter_signature(),
            semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS),
        target=FrozenTargetPotential(lambda q: -target.log_prob_and_grad(q)[0],
            identity="m17-exact-potential-" + target.adapter_signature()), target_scope="inference_validation")
    config = replace(_config(), parameter_dimension=target.parameter_dim, verification_results=64,
        max_leapfrog_steps=3, target_scope="inference_validation",
        acceptance_policy=FourChainMeanBandAcceptancePolicy(overall_band=(.65, .75), per_chain_band=(.55, .85)))
    search = HMCControllerConfig(primary_l_grid=(2, 3), initial_epsilon=.8,
        pilot_enabled=False, refinement_rounds=0, evidence_rungs=(1,),
        max_candidates=8, total_budget_units=20, repair_reserve_units=4)
    first = tune_hmc_kernel(adapter=target, initial_position=initial_starts(target, "dispersed"),
        parameter_scales=tf.ones([target.parameter_dim], tf.float64), config=config,
        search_config=search, runner_binding=binding, output_dir=tmp_path, max_work_items=1)
    assert len(first.result.observations) == 1
    resumed = resume_position_field_candidate_tuning(tmp_path / "controller_checkpoint.json",
        adapter=target, runner_binding=binding)
    assert resumed.result.completion_status == "complete"
    assert resumed.result.observations[0] == json.loads(json.dumps(first.result.observations[0]))
    assert not resumed.numerical_handoff_authority
    assert all(not row["observation"]["numerical_handoff_authority"] for row in resumed.result.observations)
    with pytest.raises((ValueError, TypeError), match="binding|numerical|verified"):
        build_retained_bound_hmc_archive_runner_from_candidate_set_result(
            candidate_set_result=resumed.result,
            candidate_id=resumed.result.candidates[0].candidate_id,
            retained_binding=binding)


@pytest.mark.parametrize("target_id,data", [("gaussian", None), ("beta_binomial", [7, 12]),
                                           ("lgssm_location", [1., -1., .5]), ("banana", None)])
def test_position_field_directional_pilots_reach_own_fresh_verification(tmp_path, target_id, data):
    from bayesfilter.inference import (FourChainMeanBandAcceptancePolicy, FrozenPositionOnlyForce,
        FrozenTargetPotential, DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
        bind_neural_force_hmc_tuning_runner, tune_hmc_kernel, resume_position_field_candidate_tuning,
        HMCControllerConfig)
    from bayesfilter.testing.inference_validation.designs import seed_for
    from tests.test_hmc_tuning_dispatch import _config

    target = ValidationTarget(target_id, data=data, jit_compile=False)
    binding = bind_neural_force_hmc_tuning_runner(
        force=FrozenPositionOnlyForce(lambda q: -.8 * target.log_prob_and_grad(q)[1],
            identity="m17-force-" + target.adapter_signature(),
            semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS),
        target=FrozenTargetPotential(lambda q: -target.log_prob_and_grad(q)[0],
            identity="m17-potential-" + target.adapter_signature()), target_scope="inference_validation")
    config = replace(_config(), parameter_dimension=target.parameter_dim,
        mass_window_results=(32, 64), step_adaptation_results=64, verification_results=128,
        max_leapfrog_steps=25, initial_step_size=.5, verification_repair_rounds=3,
        seed=seed_for(2026092197, "position-field", "cpu_reference", target_id),
        target_scope="inference_validation",
        acceptance_policy=FourChainMeanBandAcceptancePolicy(overall_band=(.65, .75), per_chain_band=(.55, .85)))
    search = HMCControllerConfig(primary_l_grid=(3, 5, 9, 13, 18, 25), initial_epsilon=.5,
        pilot_enabled=True, refinement_rounds=1, max_candidates=100,
        total_budget_units=300, repair_reserve_units=40)
    tune_hmc_kernel(adapter=target, initial_position=initial_starts(target, "dispersed"),
        parameter_scales=tf.ones([target.parameter_dim], tf.float64), config=config,
        search_config=search, runner_binding=binding, output_dir=tmp_path, max_work_items=1)
    resumed = resume_position_field_candidate_tuning(tmp_path / "controller_checkpoint.json",
        adapter=target, runner_binding=binding)
    payload = read_json(tmp_path / "candidate_set_result.json")
    assert not check_inventory(payload)["failures"]
    assert resumed.result.completion_status == "complete"
    assert resumed.result.verified_candidate_ids
    assert {row["stage"] for row in resumed.result.observations} == {"pilot", "measurement", "verification"}
    for candidate in resumed.result.verified_candidate_ids:
        observations = [row for row in resumed.result.observations if row["candidate_id"] == candidate]
        assert {"measurement", "verification"} <= {row["stage"] for row in observations}
        streams = [row["observation"]["seed_lineage"] for row in observations]
        assert len({tuple(seed) for seed in streams}) == len(streams)
        assert all(not row["observation"]["numerical_handoff_authority"] for row in observations)
