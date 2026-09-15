"""Independent CPU reference and lifecycle regressions; no sampler promotion."""
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.stats import norm, rankdata
import tensorflow as tf

from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCCandidateSetScope, HMCControllerConfig, HMCTuningCandidateSetController,
)
from bayesfilter.inference.hmc_candidate_set_artifacts import (
    candidate_set_result_payload, _validate_result_payload, require_verified_member,
)
from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
from bayesfilter.inference.hmc_posterior_diagnostics import rank_normalized_split_rhat
from bayesfilter.inference.hmc_precision import (
    HMCPrecisionTarget, HMCPrecisionPolicy, mean_precision, precision_report,
)
from bayesfilter.inference.hmc_posterior_assessment import (
    HMCPosteriorAssessmentPolicy, assess_posterior, validate_sequential_seeds,
)
from tests.test_hmc_candidate_set_execution import GaussianTarget, make_binding, execution_config
from bayesfilter.inference import (
    tune_hmc_kernel, build_retained_bound_hmc_archive_runner_from_candidate_set_result,
    load_hmc_candidate_retained_runner, load_numerical_tuning_checkpoint,
)
from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig


def scope():
    return HMCCandidateSetScope(scope_id="overall", search_id="repair", target_signature="synthetic",
        mass_signature="identity", coordinate_system="ordinary", start_bank_signature="starts",
        warmup_protocol="none", epsilon_domain=(.05, 2.), repair_factor=2., max_repairs_per_family=2)


def test_released_reservation_finishes_without_extra_resume():
    def observe(work, candidate):
        if candidate.leapfrog_steps == 3:
            if candidate.parent_candidate_id is None:
                return {"decision": "repair_step_lower", "acceptance": .2}
            if work.stage == "measurement" and work.evidence_rung < 2:
                return {"decision": "inconclusive_evidence", "acceptance": .7}
        return {"decision": "passed", "acceptance": .7}
    result = HMCTuningCandidateSetController(scope(), HMCControllerConfig(primary_l_grid=(3, 5),
        epsilon_by_l=((3, (.5,)), (5, (.5,))), total_budget_units=7,
        repair_reserve_units=3, candidate_reserve_units=3, evidence_rungs=(1, 2, 4))).run(observe)
    assert result.completion_status == "complete"
    assert result.budget_used_units == 7
    assert len(result.verified_candidate_ids) == 2


def test_verified_repair_then_shared_invalidity_round_trips():
    def observe(work, candidate):
        if candidate.leapfrog_steps == 3 and candidate.parent_candidate_id is None:
            return {"decision": "repair_step_lower", "acceptance": .2}
        if candidate.leapfrog_steps == 5 and work.stage == "verification":
            return {"decision": "failed", "evidence_validity": "shared_execution_invalid"}
        return {"decision": "passed", "acceptance": .7}
    result = HMCTuningCandidateSetController(scope(), HMCControllerConfig(primary_l_grid=(3, 5),
        epsilon_by_l=((3, (.5,)), (5, (.5,))), total_budget_units=15, repair_reserve_units=3)).run(observe)
    payload = json.loads(json.dumps(candidate_set_result_payload(result)))
    _validate_result_payload(payload)
    restored = HMCTuningCandidateSetController.from_result_payload(payload).result()
    assert restored.completion_status == "shared_invalidity"
    assert not restored.verified_candidate_ids
    assert any(a.qualified_repair_status == "executed_and_verified" for a in restored.repair_actions)
    with pytest.raises(ValueError):
        require_verified_member(payload, scope_id=scope().scope_id, candidate_id=result.candidates[-1].candidate_id)


def reference_rhat(x):
    half = len(x) // 2
    x = np.concatenate((x[:half], x[-half:]), axis=1)
    def component(v):
        ranks = rankdata(v, method="average").reshape(v.shape)
        z = norm.ppf((ranks - 3 / 8) / (v.size + 1 / 4))
        w = z.var(axis=0, ddof=1).mean()
        return np.sqrt(((half - 1) / half * w + z.mean(axis=0).var(ddof=1)) / w)
    return max(component(x), component(np.abs(x - np.median(x))))


@pytest.mark.parametrize("count", [64, 65])
def test_both_rhat_interfaces_match_published_formula(count):
    x = np.sin(np.arange(count)[:, None] * .7 + np.arange(4)) + np.arange(4) * .3
    values = tf.constant(x[..., None], tf.float64)
    expected = reference_rhat(x)
    assert rank_normalized_split_rhat_summary(values)["rhat"][0] == pytest.approx(expected, abs=1e-12)
    assert float(rank_normalized_split_rhat(tf.transpose(values, (1, 0, 2)))["maximum"][0]) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("method", ["batch_means", "lugsail"])
def test_batch_mcse_matches_independent_chain_combination(method):
    x = np.random.default_rng(915).normal(size=(303, 4, 2))
    def bm(b):
        a = len(x) // b
        return b * x[:a*b].reshape(a, b, 4, 2).mean(axis=1).var(axis=0, ddof=1)
    expected_lrv = bm(15) if method == "batch_means" else 2 * bm(15) - bm(5)
    report = mean_precision(tf.constant(x), method=method, batch_size=15, min_batches=10, jit_compile=False)
    np.testing.assert_allclose(report["mcse"], np.sqrt(expected_lrv.sum(axis=0)/(16*303)), rtol=1e-12)
    assert report["unused_terminal_draws"] == 3


def test_constant_and_underbatched_samples_do_not_claim_precision():
    policy = HMCPrecisionPolicy((HMCPrecisionTarget("x", mcse_absolute_max=1.),),
        method="lugsail", jit_compile=False)
    report = precision_report(tf.ones((64, 4, 1), tf.float64), ("x",), policy)
    assert not report["passed"] and report["targets"][0]["mcse"] is None
    x = tf.constant(np.random.default_rng(4).normal(size=(64, 4, 1)))
    report = precision_report(x, ("x",), policy)
    assert report["batch_metadata"]["unavailable_reason"] == "insufficient_complete_batches"


def test_small_rhat_and_passing_callback_cannot_skip_precision_or_core():
    x = tf.constant(np.random.default_rng(8).normal(size=(512, 4, 1)))
    policy = HMCPosteriorAssessmentPolicy(precision=HMCPrecisionPolicy((
        HMCPrecisionTarget("x", mcse_absolute_max=1e-8),)))
    report = assess_posterior(x, ("x",), policy=policy, stage="retained",
        rhat={"passed": True}, extra={"passed": True})
    assert not report["passed"] and report["precision"]["status"] == "precision_insufficient"
    report = assess_posterior(x, ("x",), policy=None, stage="retained",
        rhat={"passed": False}, extra={"passed": True})
    assert not report["passed"]


def seq_config(member=None, **overrides):
    values = dict(step_size=1.3 if member is None else member.step_size,
        num_leapfrog_steps=3 if member is None else member.num_leapfrog_steps,
        warmup_seed=(20260915, 17), retained_seed=(20260915, 19), jit_compile=False,
        warmup_chunk_results=16, warmup_min_results=16, warmup_check_window_results=16,
        warmup_max_results=32, retained_chunk_results=16, retained_min_results=16, retained_max_results=32)
    return SequentialNeuTraHMCConfig(**(values | overrides))


def test_complete_seed_schedule_rejects_later_overlap_and_tuning_overlap():
    config = seq_config(warmup_seed=(17, 0), retained_seed=(17, 1009))
    with pytest.raises(ValueError, match="disjoint"):
        validate_sequential_seeds(config)
    with pytest.raises(ValueError, match="disjoint"):
        validate_sequential_seeds(seq_config(), forbidden={(20260915, 17 + 2018)})
    with pytest.raises(ValueError, match="int32"):
        validate_sequential_seeds(seq_config(warmup_seed=(17, 2**31 - 2)))


class ScalarGaussian(GaussianTarget):
    def log_prob_and_grad(self, theta):
        tf.debugging.assert_rank(theta, 1)
        return super().log_prob_and_grad(theta)


class NoTelemetryGaussian(GaussianTarget):
    target_status_telemetry = None


@pytest.mark.parametrize("target", [ScalarGaussian, NoTelemetryGaussian])
def test_actual_verified_member_reaches_posterior_with_original_contract(target, monkeypatch):
    adapter = target()
    binding = make_binding(target=adapter, config=execution_config(target_status_trace_policy="none"))
    run = tune_hmc_kernel(adapter=adapter, initial_position=binding.initial_active_state,
        config=HMCControllerConfig(primary_l_grid=(3,), epsilon_by_l=((3, (1.3,)),),
            total_budget_units=10, repair_reserve_units=3, evidence_rungs=(1,)),
        candidate_set_adapter=binding.typed_adapter)
    assert run.result.verified_candidate_ids
    member = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=run.result, candidate_id=run.result.verified_candidate_ids[0], retained_binding=binding)
    # This test checks runner/telemetry integration, with convergence delegated
    # to separately tested diagnostics. Force readiness to exercise retention.
    from bayesfilter.inference import neutra_hmc
    monkeypatch.setattr(neutra_hmc, "_sequential_rhat",
        lambda *_args, **kwargs: {"passed": True, "rhat_threshold": kwargs["rhat_max"]})
    result = member.run_sequential(config=seq_config(member), parameter_names=("x", "y"))
    assert result["retained_results_per_chain"] >= 16
    assert result["warmup_results_per_chain"] >= 16
    assert not result["hard_vetoes"]
    assert result["warmup_checks"][0]["health"]["target_status_trace_policy"] == "none"


class DomainTarget(GaussianTarget):
    def classify_target_exception(self, error):
        return isinstance(error, tf.errors.InvalidArgumentError) and "declared domain" in str(error)


def test_declared_candidate_exception_preserves_peer_and_reload(tmp_path, monkeypatch):
    target = DomainTarget()
    binding = make_binding(target=target)
    original = binding._run
    def fail_one(candidate, *args):
        if candidate.leapfrog_steps == 2:
            raise tf.errors.InvalidArgumentError(None, None, "declared domain")
        return original(candidate, *args)
    monkeypatch.setattr(binding, "_run", fail_one)
    run = tune_hmc_kernel(adapter=target, initial_position=binding.initial_active_state,
        config=HMCControllerConfig(primary_l_grid=(2, 3), epsilon_by_l=((2, (1.3,)), (3, (1.3,))),
            total_budget_units=10, repair_reserve_units=3, evidence_rungs=(1,)),
        candidate_set_adapter=binding.typed_adapter, output_dir=tmp_path / "tune")
    assert run.result.completion_status == "complete" and len(run.result.verified_candidate_ids) == 1
    failures = [r for r in run.result.verification_receipts if r.evidence_validity == "candidate_data_invalid"]
    assert len(failures) == 1 and failures[0].acceptance is None and not failures[0].repair_eligible
    restored, controller = load_numerical_tuning_checkpoint(tmp_path / "tune/tuning_checkpoint.json", adapter=target)
    member = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=controller.result(), candidate_id=run.result.verified_candidate_ids[0], retained_binding=restored)
    loaded = load_hmc_candidate_retained_runner(member.export(tmp_path / "member.json"), adapter=target)
    assert loaded.candidate == member.candidate


def test_quantile_mcse_matches_independent_order_statistic_reference():
    from scipy.stats import beta
    from bayesfilter.inference.hmc_precision import quantile_precision
    from bayesfilter.inference.hmc_diagnostic_math import _cross_chain_ess
    x = np.random.default_rng(919).exponential(size=(601, 4, 1))
    report = quantile_precision(tf.constant(x), .9)
    q = np.quantile(x, .9)
    split = np.concatenate((x[:300], x[-300:]), axis=1)
    ess = float(_cross_chain_ess(tf.constant((split <= q).astype(float)))[0])
    bounds = beta.ppf(norm.cdf([-1., 1.]), ess*.9+1, ess*.1+1)
    indices = [max(int(np.floor(bounds[0]*x.size)), 1)-1,
               min(int(np.ceil(bounds[1]*x.size)), x.size)-1]
    expected = np.diff(np.sort(x.ravel())[indices])[0]/2
    assert float(report['mcse'][0]) == pytest.approx(expected, abs=1e-12)
    assert not bool(quantile_precision(tf.ones((64,4,1), tf.float64), .9)['valid'][0])


def test_negative_lugsail_is_reported_not_clipped():
    # Each complete large batch has exactly zero mean, but short batches vary.
    x = np.tile(np.array([1., 1., 1., -1., -1., -1.]), 40)
    draws = tf.constant(np.broadcast_to(x[:, None, None], (240, 4, 1)))
    policy = HMCPrecisionPolicy((HMCPrecisionTarget('x', mcse_absolute_max=1.),),
        method='lugsail', batch_size=6, jit_compile=False)
    report = precision_report(draws, ('x',), policy)
    assert not report['passed']
    assert report['batch_metadata']['per_chain_long_run_variance'][0][0] < 0
    json.dumps(report, allow_nan=False)


def test_declared_functionals_and_rare_events_receive_core_checks():
    draws = tf.constant(np.random.default_rng(19).normal(size=(256,4,1)))
    policy = HMCPosteriorAssessmentPolicy(quantities_id='rare-event-v1', precision=HMCPrecisionPolicy((
        HMCPrecisionTarget('rare_event', mcse_absolute_max=.01),)))
    report = assess_posterior(draws, ('x',), policy=policy, stage='retained',
        rhat={'passed': True, 'rhat_threshold': 1.01},
        quantities_fn=lambda x: {'rare_event': x[:,:,0] > 10.})
    assert not report['passed']
    assert report['precision']['targets'][0]['mcse'] is None
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match='requires quantities_fn'):
        assess_posterior(draws, ('x',), policy=policy, stage='retained', rhat={'passed': True})


def test_warmup_persistence_precision_cap_and_callback_veto(monkeypatch):
    from tests.test_neutra_hmc import _fake_programs, _script_rhat
    from bayesfilter.inference import neutra_hmc
    _fake_programs(monkeypatch)
    _script_rhat(monkeypatch, (True, True, True, True))
    cfg = seq_config(warmup_chunk_results=4, warmup_min_results=4, warmup_check_window_results=4,
        warmup_max_results=8, retained_chunk_results=4, retained_min_results=4, retained_max_results=8,
        assessment_policy=HMCPosteriorAssessmentPolicy(warmup_consecutive_checks=2,
            precision=HMCPrecisionPolicy((HMCPrecisionTarget('x', mcse_absolute_max=1e-10),))))
    result = neutra_hmc.run_sequential_neutra_hmc(adapter=object(),
        initial_state=tf.zeros((4,1),tf.float64), parameter_names=('x',), config=cfg,
        retained_diagnostic_fn=lambda _: {'passed': True})
    assert result['warmup_results_per_chain'] == 8
    assert result['retained_cap_hit'] and not result['passed']
    assert result['precision_status'] == 'precision_insufficient'
    _script_rhat(monkeypatch, (True, True, True))
    result = neutra_hmc.run_sequential_neutra_hmc(adapter=object(),
        initial_state=tf.zeros((4,1),tf.float64), parameter_names=('x',), config=cfg,
        retained_diagnostic_fn=lambda _: {'passed': True, 'hard_vetoes': ['reference_mismatch']})
    assert result['hard_vetoes'] == ('reference_mismatch',)
    assert result['retained_results_per_chain'] == 4


@pytest.fixture
def overall_member():
    adapter = GaussianTarget()
    binding = make_binding(target=adapter)
    run = tune_hmc_kernel(adapter=adapter, initial_position=binding.initial_active_state,
        config=HMCControllerConfig(primary_l_grid=(3,), epsilon_by_l=((3,(1.3,)),),
            total_budget_units=10, repair_reserve_units=3, evidence_rungs=(1,)),
        candidate_set_adapter=binding.typed_adapter)
    return build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=run.result, candidate_id=run.result.verified_candidate_ids[0], retained_binding=binding)


def test_member_sequential_checkpoint_replays_without_native_calls(overall_member, tmp_path, monkeypatch):
    from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
    from bayesfilter.inference import neutra_hmc, run_hmc_posterior
    member = overall_member
    monkeypatch.setattr(neutra_hmc, '_sequential_rhat',
        lambda *_args, **kwargs: {'passed': True, 'rhat_threshold': kwargs['rhat_max']})
    config = seq_config(member)
    with DurableTensorCheckpoint(tmp_path/'chunks', {'member': member.member_hash}) as store:
        first = run_hmc_posterior(member=member, config=config, parameter_names=('x','y'), checkpoint_store=store)
    def unexpected(*_args):
        raise AssertionError('completed chunk was rerun')
    monkeypatch.setattr(member._binding, '_run', unexpected)
    with DurableTensorCheckpoint(tmp_path/'chunks', {'member': member.member_hash}) as store:
        replay = run_hmc_posterior(member=member, config=config, parameter_names=('x','y'), checkpoint_store=store)
    np.testing.assert_array_equal(first['private_retained_raw'], replay['private_retained_raw'])
    assert replay['precision_status'] == 'precision_not_requested'
    with DurableTensorCheckpoint(tmp_path/'chunks', {'member': member.member_hash}) as store:
        from bayesfilter.runtime.durable_tensor_checkpoint import CheckpointError
        with pytest.raises(CheckpointError):
            run_hmc_posterior(member=member, config=replace(config, assessment_policy=HMCPosteriorAssessmentPolicy(
                warmup_consecutive_checks=2)), parameter_names=('x','y'), checkpoint_store=store)


def test_compact_shared_exports_cache_and_corruption(overall_member, tmp_path, monkeypatch):
    from bayesfilter.inference.hmc_candidate_set_checkpoint import write_numerical_tuning_checkpoint
    member = overall_member
    binding = member._binding
    member._validate()  # Fill the content-addressed numerical-analysis cache.
    def unexpected(*_args):
        raise AssertionError('unchanged numerical evidence reanalyzed')
    monkeypatch.setattr(binding, 'analyze', unexpected)
    paths = [member.export(tmp_path/f'member-{i}.json') for i in range(3)]
    bundles = list(tmp_path.glob('evidence-*.json'))
    assert len(bundles) == 1
    assert paths[0].stat().st_size < bundles[0].stat().st_size / 5
    loaded = load_hmc_candidate_retained_runner(paths[0], adapter=GaussianTarget())
    assert loaded.member_hash == member.member_hash
    write_numerical_tuning_checkpoint(binding, HMCTuningCandidateSetController.from_result_payload(member._result).result(), tmp_path/'checkpoint')
    evidence_files = list((tmp_path/'checkpoint/numerical_evidence').glob('*.json'))
    times = [p.stat().st_mtime_ns for p in evidence_files]
    write_numerical_tuning_checkpoint(binding, HMCTuningCandidateSetController.from_result_payload(member._result).result(), tmp_path/'checkpoint')
    assert times == [p.stat().st_mtime_ns for p in evidence_files]
    evidence_files[0].write_text('{}')
    with pytest.raises(ValueError, match='collision'):
        write_numerical_tuning_checkpoint(binding, HMCTuningCandidateSetController.from_result_payload(member._result).result(), tmp_path/'checkpoint')
    numerical = next(iter(binding._evidence.values()))
    numerical['analysis']['acceptance'] = -1.
    with pytest.raises(ValueError, match='corrupt'):
        member._validate()


def test_missing_mode_is_not_detected_by_small_mcse():
    # All chains sample just one component of an equally weighted +/-20 mixture.
    # The diagnostics cannot know about the unvisited component. This is an
    # explicit detection-limit counterexample, not a convergence requirement.
    x = tf.constant(np.random.default_rng(111).normal(-20., 1., (2048,4,1)))
    policy = HMCPosteriorAssessmentPolicy(precision=HMCPrecisionPolicy((
        HMCPrecisionTarget('x', mcse_absolute_max=.1),)))
    result = assess_posterior(x, ('x',), policy=policy, stage='retained',
        rhat=rank_normalized_split_rhat_summary(x))
    assert result['passed']
    assert abs(result['precision']['targets'][0]['estimate']) > 19.


def test_posterior_summary_preserves_consumer_fields_without_overriding_core():
    draws = tf.constant(np.random.default_rng(222).normal(size=(64,4,1)))
    result = assess_posterior(draws, ('x',), policy=None, stage='retained',
        rhat={'passed':False}, extra={'passed':True,'max_rhat':1.001,'reference_distance':.02})
    assert result['max_rhat'] == 1.001 and result['reference_distance'] == .02
    assert not result['passed'] and not result['modern_rhat']['passed']


def test_retained_ancestry_can_exceed_python_recursion_depth(tmp_path, monkeypatch):
    # Ancestry reader regression: numerical-health arithmetic is separately
    # tested with actual traces. Keep this fixture about links/endpoints/seeds.
    import types
    from bayesfilter.inference.hmc_candidate_set_retained import HMCCandidateRetainedRunner, RETAINED_ARCHIVE_SCHEMA
    from bayesfilter.inference.hmc_candidate_set_execution import _tensor_payload, _trace_payload
    from bayesfilter.inference.hmc_candidate_set_tuning import _sha256
    candidate = types.SimpleNamespace(payload=lambda: {'candidate_id':'ancestry-fixture'})
    binding = types.SimpleNamespace(binding_hash='binding', health_failures=lambda *_: (),
        position_samples=lambda x:x)
    runner = object.__new__(HMCCandidateRetainedRunner)
    runner.candidate, runner._binding, runner.member_hash = candidate, binding, 'member'
    runner.initial_active_state = tf.zeros((2,1),tf.float64)
    monkeypatch.setattr(runner,'_tuning_seeds',lambda:set())
    previous = None
    state = runner.initial_active_state
    history = []
    for i in range(1050):  # Beyond Python's usual recursion ceiling of 1000.
        final = state+1.
        seed = [500,i]
        history.append(seed)
        body = {'schema':RETAINED_ARCHIVE_SCHEMA,'member_hash':'member','binding_hash':'binding',
            'candidate':candidate.payload(),'health_failures':[], 'num_results':1,
            'initial_active_state':_tensor_payload(state),'final_active_state':_tensor_payload(final),
            'active_samples':_tensor_payload(final[None]),'position_samples':_tensor_payload(final[None]),
            'trace':_trace_payload({}),'seed':seed,'seed_history':list(history),'predecessor':previous}
        digest = _sha256(body)
        path = tmp_path/f'{i}.json'
        path.write_text(json.dumps({**body,'content_hash':digest}))
        previous = {'path':str(path),'content_hash':digest,'final_active_state_hash':body['final_active_state']['sha256']}
        state = final
    payload, endpoint = runner._archive(path)
    np.testing.assert_array_equal(endpoint, np.full((2,1),1050.))
    assert len(payload['seed_history']) == 1050
