"""Finite-trial health, counting and recovery checks (independent references)."""
import math

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_acceptance_statistics import complete_trial_scores
from bayesfilter.inference import hmc_verification as verification
from tests.test_hmc_acceptance_protocol import policy
from tests.test_hmc_verification import _moving_samples


def _replicated_setup(**changes):
    from tests.test_hmc_candidate_set_execution import execution_config, make_binding
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig
    p = policy(base_repetitions=1, max_repetitions=2, trial_num_results=65,
               discarded_prefix=3, max_candidates=2, **changes)
    execution = execution_config(measurement_num_results=65, verification_num_results=65,
        num_warmup_steps=3, acceptance_policy=p, chunk_max_results=34, chain_mode="batched")
    binding = make_binding(config=execution)
    search = HMCControllerConfig(primary_l_grid=(2,), epsilon_by_l=((2, (1.3,)),),
        total_budget_units=10, repair_reserve_units=1, max_candidates=2,
        evidence_rungs=(1, 2), replicated_acceptance_policy=p)
    return binding, search


def test_public_trial_rungs_add_repetitions_preserve_starts_and_recompute(tmp_path):
    from bayesfilter.inference.hmc_tuning_dispatch import tune_hmc_kernel
    from bayesfilter.inference.hmc_candidate_set_checkpoint import load_numerical_tuning_checkpoint
    from bayesfilter.inference.hmc_candidate_set_execution import _tensor_payload
    binding, search = _replicated_setup()
    run = tune_hmc_kernel(adapter=binding._base_adapter, initial_position=binding.initial_active_state,
        config=search, candidate_set_adapter=binding.typed_adapter, output_dir=tmp_path)
    result = run.result
    works = result.work_items
    assert [w.trial_range for w in works] == [(0, 1), (1, 2)]
    assert all(w.stage == "measurement" for w in works)
    assert set(result.candidate_states.values()) == {"inconclusive_at_cap"}
    rows = list(binding._evidence.values())
    assert [len(row["trials"]) for row in rows] == [1, 2]
    assert rows[0]["trials"][0] == rows[1]["trials"][0]
    assert rows[0]["seed"] == rows[1]["seed"]
    initial = _tensor_payload(binding.initial_active_state)
    assert all(t["chunks"][0]["initial_state"] == initial for t in rows[-1]["trials"])
    assert len({tuple(c["seed"]) for row in rows for c in row["chunks"]}) == 4
    # 2 trials * (3 discarded + 65 measured) * 4 starts * (L+1).
    charged = sum(e["gradient_work"] for e in result.accounting_events if e["event"] == "numerical_chunk_charged")
    assert charged == 2*68*4*3
    loaded, controller = load_numerical_tuning_checkpoint(tmp_path/"tuning_checkpoint.json", adapter=binding._base_adapter)
    assert controller.result().candidate_states == result.candidate_states
    assert loaded._evidence == binding._evidence


def test_replication_design_mismatch_fails_before_candidate_work():
    from dataclasses import replace
    from bayesfilter.inference.hmc_tuning_dispatch import tune_hmc_kernel
    binding, search = _replicated_setup()
    with pytest.raises(ValueError, match="same replicated"):
        tune_hmc_kernel(adapter=binding._base_adapter, initial_position=binding.initial_active_state,
            config=replace(search, replicated_acceptance_policy=None),
            candidate_set_adapter=binding.typed_adapter)


def test_interrupted_new_trial_resumes_same_stream_without_reusing_information(tmp_path, monkeypatch):
    from bayesfilter.inference.hmc_candidate_set_adapters import run_typed_hmc_candidate_set
    from bayesfilter.inference.hmc_candidate_set_checkpoint import resume_hmc_candidate_set_tuning
    binding, search = _replicated_setup()
    original = binding._run
    calls = []
    def interrupt(candidate, state, count, seed):
        calls.append(tuple(seed))
        if len(calls) == 3:
            raise tf.errors.UnavailableError(None, None, "injected at trial reset")
        return original(candidate, state, count, seed)
    monkeypatch.setattr(binding, "_run", interrupt)
    paused = run_typed_hmc_candidate_set(binding.typed_adapter, search, output_dir=tmp_path)
    assert len(binding._evidence) == 1
    assert paused.result.completion_status == "paused_infrastructure"
    resumed = resume_hmc_candidate_set_tuning(tmp_path/"tuning_checkpoint.json", adapter=binding._base_adapter)
    events = [e for e in resumed.result.accounting_events if e["event"] == "numerical_chunk_charged"]
    assert tuple(events[2]["seed"]) == tuple(events[3]["seed"]) == calls[2]
    assert sum(e["gradient_work"] for e in events) == (2*68+34)*4*3
    evidence = list(resumed.adapter._execution_binding._evidence.values())[-1]
    assert len(evidence["trials"]) == 2


def test_position_field_uses_same_trial_driver_and_replays_its_checkpoint(tmp_path):
    from dataclasses import replace
    from tests.test_hmc_tuning_dispatch import _Adapter, _binding, _config
    from bayesfilter.inference import tune_hmc_kernel, resume_position_field_candidate_tuning
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig
    from bayesfilter.inference.hmc_candidate_set_execution import HMCCandidateExecutionConfig
    p = policy(base_repetitions=1, max_repetitions=2, max_candidates=2,
               trial_num_results=65, discarded_prefix=3)
    cfg = replace(_config(), verification_results=65, max_leapfrog_steps=5)
    execution = HMCCandidateExecutionConfig(measurement_num_results=65, verification_num_results=65,
        num_warmup_steps=3, seed=(1002, 8), acceptance_policy=p,
        target_status_trace_policy=cfg.target_status_trace_policy, use_xla=False,
        non_xla_reason="CPU mechanics replication/recovery regression", chunk_max_results=34)
    search = HMCControllerConfig(primary_l_grid=(3,), epsilon_by_l=((3, (1.3,)),),
        total_budget_units=10, repair_reserve_units=1, max_candidates=2,
        evidence_rungs=(1, 2), replicated_acceptance_policy=p)
    first = tune_hmc_kernel(adapter=_Adapter(), initial_position=tf.zeros([4, 2], tf.float64),
        parameter_scales=tf.ones([2], tf.float64), config=cfg, search_config=search,
        execution_config=execution, runner_binding=_binding(), output_dir=tmp_path, max_work_items=1)
    runtime = first.adapter._checkpoint_runtime
    assert len(runtime._evidence) == 1
    resumed = resume_position_field_candidate_tuning(tmp_path/"controller_checkpoint.json",
        adapter=_Adapter(), runner_binding=_binding())
    assert not resumed.numerical_handoff_authority
    evidence = list(resumed.adapter._checkpoint_runtime._evidence.values())[-1]
    assert len(evidence["trials"]) == 2
    assert [t["ordinal"] for t in evidence["trials"]] == [0, 1]


def test_evidence_saved_before_receipt_is_replayed_without_another_trial(tmp_path, monkeypatch):
    from bayesfilter.inference.hmc_candidate_set_adapters import run_typed_hmc_candidate_set
    from bayesfilter.inference.hmc_candidate_set_checkpoint import resume_hmc_candidate_set_tuning
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCTuningCandidateSetController, HMCInfrastructureFailure
    binding, search = _replicated_setup()
    original = HMCTuningCandidateSetController._apply_observation
    calls = []
    def interrupt(self, work, observation):
        calls.append(work.work_item_id)
        if len(calls) == 1:
            raise HMCInfrastructureFailure("injected after numerical persistence")
        return original(self, work, observation)
    monkeypatch.setattr(HMCTuningCandidateSetController, "_apply_observation", interrupt)
    first = run_typed_hmc_candidate_set(binding.typed_adapter, search, output_dir=tmp_path)
    digest = next(iter(binding._evidence))
    assert first.result.completion_status == "paused_infrastructure"
    second = resume_hmc_candidate_set_tuning(tmp_path/"tuning_checkpoint.json", adapter=binding._base_adapter)
    assert digest in second.adapter._execution_binding._evidence
    assert len(second.adapter._execution_binding._evidence) == 2
    charged = sum(e["gradient_work"] for e in second.result.accounting_events if e["event"] == "numerical_chunk_charged")
    assert charged == 2*68*4*3


def test_trial_evidence_tampering_cannot_survive_recomputed_hashes():
    import copy
    from bayesfilter.inference.hmc_candidate_set_adapters import run_typed_hmc_candidate_set
    binding, search = _replicated_setup()
    run_typed_hmc_candidate_set(binding.typed_adapter, search)
    original = list(binding._evidence.values())[-1]
    for mutate in (
        lambda row: row["trials"][1].update(seed=row["trials"][0]["seed"]),
        lambda row: row["trials"][1]["scores"].update(start_scores=[.7]*4),
        lambda row: row.update(samples=row["trials"][0]["samples"]),
        lambda row: row.update(evidence_rungs=[1, 4]),
        lambda row: row["chunks"][0].update(trial_ordinal=0),
    ):
        changed = copy.deepcopy(original)
        mutate(changed)
        with pytest.raises(ValueError):
            binding.evidence_analysis(changed)


def test_replay_reuses_decoded_chunks_only_within_one_checked_analysis(monkeypatch):
    import copy
    from bayesfilter.inference.hmc_candidate_set_adapters import run_typed_hmc_candidate_set
    from bayesfilter.inference import hmc_candidate_set_execution as execution
    binding, search = _replicated_setup()
    run_typed_hmc_candidate_set(binding.typed_adapter, search)
    numerical = next(iter(binding._evidence.values()))
    binding._analysis_cache.clear()
    calls = []
    original = execution._trace_from_payload
    def decoded(raw):
        if 'log_accept_ratio' in raw:  # Exclude recursive status-telemetry maps.
            calls.append(1)
        return original(raw)
    monkeypatch.setattr(execution, '_trace_from_payload', decoded)
    assert execution._json_copy(binding.evidence_analysis(numerical)) == numerical['analysis']
    assert len(calls) == len(numerical['chunks'])
    changed = copy.deepcopy(numerical)
    changed['chunks'][0]['trace']['log_accept_ratio']['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        binding.evidence_analysis(changed)


def test_duplicate_trial_stream_fails_before_it_can_supply_more_evidence(monkeypatch):
    from bayesfilter.inference.hmc_candidate_set_adapters import run_typed_hmc_candidate_set
    from bayesfilter.inference import hmc_acceptance_trials as trials
    binding, search = _replicated_setup()
    monkeypatch.setattr(trials, "trial_seed", lambda root, ordinal: (177, 913))
    run = run_typed_hmc_candidate_set(binding.typed_adapter, search)
    assert run.result.completion_status == "shared_invalidity"
    assert len(binding._evidence) == 1
    assert not run.result.verified_candidate_ids


def test_serial_seed_inventory_matches_the_actual_start_streams():
    from dataclasses import replace
    from tests.test_hmc_candidate_set_execution import make_binding
    binding, _ = _replicated_setup()
    serial = make_binding(config=replace(binding.config, chain_mode="serial"))
    seed = (181, 912)
    lineage = serial.trial_seed_lineage(seed)
    assert lineage[0] == seed
    assert lineage[1:] == tuple(tuple(int(x) for x in tf.random.experimental.stateless_fold_in(
        tf.constant(seed, tf.int32), i).numpy()) for i in range(4))


@pytest.mark.parametrize("mode", ["serial", "threaded"])
def test_scalar_layouts_collect_complete_trials_with_the_declared_start_streams(tmp_path, mode):
    from dataclasses import replace
    from tests.test_hmc_candidate_set_execution import make_binding
    from bayesfilter.inference import tune_hmc_kernel, load_numerical_tuning_checkpoint
    from bayesfilter.testing.acceptance_decision_models import evidence_accounting

    template, search = _replicated_setup()
    binding = make_binding(config=replace(template.config, chain_mode=mode))
    result = tune_hmc_kernel(adapter=binding._base_adapter,
        initial_position=binding.initial_active_state, config=search,
        candidate_set_adapter=binding.typed_adapter, output_dir=tmp_path).result
    assert set(result.candidate_states.values()) == {"inconclusive_at_cap"}
    trials = list(binding._evidence.values())[-1]["trials"]
    assert len(trials) == 2
    all_actual_streams = []
    for trial in trials:
        assert trial["scores"] is not None
        for chunk in trial["chunks"]:
            expected = binding.trial_seed_lineage(chunk["seed"])[1:]
            actual = tuple(tuple(s) for s in chunk["runtime"]["chain_seeds"])
            assert actual == expected
            all_actual_streams.extend(actual)
    assert len(all_actual_streams) == len(set(all_actual_streams)) == 16
    assert evidence_accounting(binding, result)["attempted_transitions"] == 2*68*4
    loaded, controller = load_numerical_tuning_checkpoint(tmp_path/"tuning_checkpoint.json",
        adapter=binding._base_adapter)
    assert loaded._evidence == binding._evidence
    assert controller.result().candidate_states == result.candidate_states


def test_measurement_extension_preserves_first_fresh_verification_budget():
    from dataclasses import replace
    from bayesfilter.inference.hmc_candidate_set_adapters import run_typed_hmc_candidate_set
    binding, search = _replicated_setup()
    # Exactly one complete measurement and one independent verification.
    search = replace(search, max_gradient_work=2*68*4*3)
    result = run_typed_hmc_candidate_set(binding.typed_adapter, search).result
    assert len(result.observations) == 1
    assert result.completion_status == "partial_budget"
    assert any(e["event"] == "work_budget_deferred" for e in result.accounting_events)
    assert len(binding._evidence) == 1


@pytest.mark.parametrize("route", ["ordinary", "fixed_transport"])
def test_replicated_gaussian_delivers_multiple_members_with_checked_exports(tmp_path, route):
    from bayesfilter.inference import tune_hmc_kernel, tune_fixed_transport_hmc_kernel
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig
    from bayesfilter.inference.hmc_candidate_set_retained import (
        build_retained_bound_hmc_archive_runner_from_candidate_set_result,
        build_retained_frozen_kernel_hmc_adapter_from_candidate_set_result,
        load_hmc_candidate_retained_runner)
    from tests.test_hmc_candidate_set_execution import GaussianTarget, execution_config, make_binding
    # Nominate interior controls using the exact equilibrium calculation for a
    # 2D unit Gaussian. If lambda>1 is the large eigenvalue of M.T@M for the
    # symplectic trajectory map M, acceptance is 2/(1+lambda): the two energy
    # components are independent chi-square(2), hence exponential. This nominates
    # eps=1.35,L=1 (~.706) and eps=1.4,L=3 (~.676). The actual finite-start trial
    # means still require measurement and independent verification below.
    for epsilon, steps in ((1.35, 1), (1.4, 3)):
        single = np.array([[1-epsilon**2/2, epsilon],
                           [-epsilon*(1-epsilon**2/4), 1-epsilon**2/2]])
        matrix = np.linalg.matrix_power(single, steps)
        expected = 2/(1+np.linalg.eigvalsh(matrix.T@matrix)[-1])
        assert .65 < expected < .75
    p = policy(trial_num_results=65, discarded_prefix=3, base_repetitions=32,
               max_repetitions=128, max_candidates=2)
    execution = execution_config(measurement_num_results=65, verification_num_results=65,
        num_warmup_steps=3, acceptance_policy=p, chunk_max_results=68, chain_mode="batched",
        use_xla=True, non_xla_reason=None, reuse_leapfrog_graphs=True)
    target = GaussianTarget()
    extra = {}
    if route == "fixed_transport":
        extra = {"mass_artifact": None, "start_coordinates": "active",
            "frozen_transport_payload": {"schema": "bayesfilter.neutra.frozen_affine_diag.v1",
                "transport_id": "replicated-identity", "dimension": 2,
                "target_signature": target.adapter_signature(), "log_jacobian_available": True,
                "shift": [0., 0.], "raw_scale": [0., 0.]}}
    binding = make_binding(target=target, config=execution, **extra)
    search = HMCControllerConfig(primary_l_grid=(1, 3), epsilon_by_l=((1, (1.35,)), (3, (1.4,))),
        total_budget_units=20, repair_reserve_units=1, max_candidates=2,
        evidence_rungs=(1, 2, 4), replicated_acceptance_policy=p)
    kwargs = dict(initial_position=binding.initial_active_state, config=search,
                  candidate_set_adapter=binding.typed_adapter)
    if route == "ordinary":
        run = tune_hmc_kernel(adapter=target, **kwargs)
        build = build_retained_bound_hmc_archive_runner_from_candidate_set_result
    else:
        run = tune_fixed_transport_hmc_kernel(base_adapter=target, fixed_transport=binding.fixed_transport, **kwargs)
        build = build_retained_frozen_kernel_hmc_adapter_from_candidate_set_result
    assert len(run.result.verified_candidate_ids) == 2, [
        (r.exact_l, r.stage, r.decision, r.acceptance, r.promotion_vetoes) for r in run.result.verification_receipts]
    for cid in run.result.verified_candidate_ids:
        member = build(candidate_set_result=run.result, candidate_id=cid, retained_binding=binding)
        restored = load_hmc_candidate_retained_runner(member.export(tmp_path/(cid.split(":")[-1]+".json")), adapter=target)
        np.testing.assert_array_equal(member.initial_active_state, restored.initial_active_state)
        charged_seeds = {tuple(e["seed"]) for e in run.result.accounting_events if e["event"] == "numerical_chunk_charged"}
        assert charged_seeds <= restored._tuning_seeds()
        matching = [e for e in binding._evidence.values() if e["candidate"]["candidate_id"] == cid
                    and e["work"]["stage"] == "verification"][-1]
        from bayesfilter.inference.hmc_candidate_set_execution import _tensor_from_payload
        np.testing.assert_array_equal(restored.initial_active_state,
                                      _tensor_from_payload(matching["trials"][-1]["samples"])[-1])


@pytest.mark.parametrize("jit", [False, True])
def test_terminal_remainder_is_measured_and_all_windows_partition_the_trial(jit):
    values = np.full((65, 4), .60)
    values[-1] = [.9, .8, .7, .6]
    result = complete_trial_scores(np.log(values), policy=policy(), jit_compile=jit)
    expected = [(64*.6 + last)/65 for last in values[-1]]
    np.testing.assert_allclose(result["start_scores"], expected, rtol=0, atol=1e-14)
    assert result["window_boundaries"] == (0, 16, 32, 48, 65)
    windows = np.asarray(result["window_scores"])
    np.testing.assert_allclose(windows @ (np.array([16, 16, 16, 17])/65), expected,
                               rtol=0, atol=1e-14)
    assert result["excluded_remainder_per_start"] == 0


@pytest.mark.parametrize("draws", [64, 66, 68])
def test_prefix_or_partial_trial_cannot_be_mislabeled_as_measured_evidence(draws):
    with pytest.raises(ValueError, match="complete"):
        complete_trial_scores(np.full((draws, 4), math.log(.7)), policy=policy())


def test_trial_health_does_not_call_legacy_acceptance_or_temporal_decisions(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy statistical decision executed in health-only path")
    monkeypatch.setattr(verification, "_chain_mean_uncertainty_interval", forbidden)
    monkeypatch.setattr(verification, "_acceptance_decision_from_summary", forbidden)
    probability = np.tile([.3, .9, .3, .9], (64, 1))
    result = verification.evaluate_hmc_trial_health(
        samples=_moving_samples(), log_accept_ratio=np.log(probability),
        is_accepted=np.ones((64, 4), bool), policy=verification.HMCAcceptancePolicy())
    assert result.evidence_validity == "valid"
    assert result.acceptance_decision == "inconclusive_evidence"
    assert not result.candidate_promotion_vetoes
    assert result.chain_mean_uncertainty_interval is None


@pytest.mark.parametrize("failure", ["divergence", "nonfinite_target", "nonfinite_state", "stuck"])
def test_trial_health_preserves_vetoes_even_with_constant_target_acceptance(failure):
    samples = _moving_samples()
    target = np.zeros((64, 4))
    kwargs = dict(native_divergence_status="available", native_divergence_count=0)
    if failure == "divergence":
        kwargs["native_divergence_count"] = 1
    elif failure == "nonfinite_target":
        target[9, 2] = np.nan
    elif failure == "nonfinite_state":
        samples[8, 1, 0] = np.nan
    else:
        samples[:] = samples[0]
    result = verification.evaluate_hmc_trial_health(samples=samples,
        log_accept_ratio=np.full((64, 4), math.log(.7)),
        is_accepted=np.ones((64, 4), bool), target_log_prob=target,
        policy=verification.HMCAcceptancePolicy(), **kwargs)
    if failure in {"nonfinite_target", "nonfinite_state"}:
        assert result.evidence_validity != "valid"
        assert result.acceptance_decision == "unavailable"
    else:
        expected = "native_divergence_positive" if failure == "divergence" else "movement_gate_failed"
        assert expected in result.candidate_promotion_vetoes
