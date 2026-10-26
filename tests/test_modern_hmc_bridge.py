"""Focused regression tests for the bounded modern-HMC bridge launcher."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


bridge = importlib.import_module("scripts.run_modern_hmc_bridge")


def test_bridge_declares_all_core_nonlinear_targets():
    assert bridge.TARGETS == (
        "LGSSM", "NL-ACC", "NL-PP-UKF", "NL-PP-SGQF", "NL-SIR", "NL-STR"
    )


def test_failure_classification_preserves_bounded_and_infrastructure_causes():
    assert bridge._failure_class(TimeoutError("wall cap")) == "bounded_execution_failure"
    assert bridge._failure_class(RuntimeError("target score failed")) == "target_or_harness_failure"


def test_search_serializes_wall_cap_and_modern_grid():
    policy = bridge._policy()
    config = bridge._search(policy, max_wall_time_seconds=17.5)
    assert config.max_wall_time_seconds == 17.5
    assert config.primary_l_grid == (2, 3)
    assert config.epsilon_by_l == ((2, (0.1,)), (3, (0.1,)))


@pytest.mark.parametrize("target_id", bridge.TARGETS)
def test_fixture_metadata_has_data_hash_and_checked_sources(target_id):
    """Every bridge target binds actual data and inspectable source closure."""
    fixture, starts, source_paths, metadata = bridge._fixture(target_id)
    assert starts.shape.rank == 2
    assert metadata["data_sha256"].startswith("sha256:")
    assert source_paths
    assert all(path for path in source_paths)
    assert metadata["source_paths_checked"] == list(source_paths)
    observations = (fixture.source_target.observations if target_id == "LGSSM"
                    else fixture.observations)
    assert observations is not None
    assert metadata['data_sha256'] == bridge._tensor_sha256(observations)
    paths = {Path(p).name for p in source_paths}
    if target_id == 'LGSSM':
        assert {'lgssm_generic_target_adapter_tf.py', 'kalman_qr_derivatives_tf.py',
                'qr_factor_tf.py', 'tf_hmc_readiness.py'} <= paths
    if target_id == 'NL-PP-SGQF':
        assert {'predator_prey_ukf_neutra_target_tf.py', 'fixed_sgqf_tf.py'} <= paths


def test_source_closure_follows_relative_and_nested_imports(tmp_path):
    from bayesfilter.testing.inference_validation.modern_bridge_audit import source_dependencies
    package = tmp_path/'bayesfilter'
    package.mkdir()
    (package/'__init__.py').write_text('')
    (package/'first.py').write_text('def f():\n    from .second import value\n')
    (package/'second.py').write_text('from bayesfilter.third import value\n')
    (package/'third.py').write_text('value = 42\n')
    paths = source_dependencies([package/'first.py'], repository_root=tmp_path)
    assert {Path(p).name for p in paths} == {'__init__.py', 'first.py', 'second.py', 'third.py'}


def test_scheduler_completion_is_not_success_when_one_candidate_is_lost():
    result = SimpleNamespace(completion_status='complete', verified_candidate_ids=['one'],
        candidates=[SimpleNamespace(candidate_id=c) for c in ('one','two')])
    work = dict(charged_gradient_work=10, expected_completed_gradient_work=10,
                unobserved_attempted_chunk_count=0)
    assert bridge._qualification_status(result, work) == 'mechanics_incomplete'
    result.verified_candidate_ids.append('two')
    assert bridge._qualification_status(result, work) == 'mechanics_verified'
    work['unobserved_attempted_chunk_count'] = 1
    assert bridge._qualification_status(result, work) == 'mechanics_incomplete'


def test_supervisor_keeps_timeout_manifest_and_existing_checkpoint(tmp_path, monkeypatch):
    output = tmp_path/'attempt'
    def timeout_worker(command, **kwargs):
        assert 0 < kwargs['timeout'] <= 10
        partial = output/'cells/tuning_checkpoint.json'
        partial.parent.mkdir()
        partial.write_text('{"partial": true}\n')
        raise bridge.subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(bridge.subprocess, 'check_output', lambda *a, **kw: 'test-commit\n')
    monkeypatch.setattr(bridge.subprocess, 'run', timeout_worker)
    args = SimpleNamespace(output=output, execution_mode='cpu_reference', targets=['LGSSM'],
                           max_wall_time_seconds=10)
    assert bridge.supervise(args) == 1
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['status'] == 'wall_budget_exhausted'
    assert manifest['partial_evidence_preserved']
    assert json.loads((output/'cells/tuning_checkpoint.json').read_text())['partial']


def test_bad_launch_retains_failure_before_tensorflow_import(tmp_path, monkeypatch):
    monkeypatch.delenv('TF_FORCE_GPU_ALLOW_GROWTH', raising=False)
    output = tmp_path/'invalid-launch'
    assert bridge.run(output, ('LGSSM',), execution_mode='cpu_reference', max_wall_time_seconds=10) == 1
    assert json.loads((output/'manifest.json').read_text())['decision'] == 'launch_failed'
    assert 'before TensorFlow import' in json.loads((output/'failure.json').read_text())['message']


def test_sir_full_status_is_accepted_and_value_score_stays_consistent():
    import tensorflow as tf
    from bayesfilter.testing.inference_validation.modern_nonlinear import make_nonlinear_comparison_target
    from bayesfilter.inference.hmc_verification import target_status_telemetry_has_failure
    fixture = make_nonlinear_comparison_target('NL-SIR', jit_compile=False)
    value, score, status = fixture.adapter.log_prob_and_grad_status(fixture.initial_state)
    assert not target_status_telemetry_has_failure(status, expected_shape=(4,))
    condition = status['innovation_condition_estimate']
    assert bool(tf.reduce_all(tf.math.is_finite(condition) & (condition >= 1)))
    assert bool(tf.reduce_all(tf.math.is_finite(value)))
    assert bool(tf.reduce_all(tf.math.is_finite(score)))


def test_frozen_geometry_preserves_adapter_scope_and_xla_capability():
    from bayesfilter.inference.posterior_adapter import value_score_capability
    fixture, starts, sources, metadata = bridge._fixture('LGSSM', use_xla=True)
    binding, _ = bridge._binding('LGSSM', fixture, starts, sources, bridge._policy(),
        data_sha256=metadata['data_sha256'], use_xla=True, non_xla_reason=None)
    assert binding._spec['target_scope'] == value_score_capability(fixture.adapter).target_scope
    assert value_score_capability(binding._active_adapter).is_accepted_full_chain_xla_diagnostic_authority


@pytest.fixture(scope='module')
def real_bridge_evidence():
    from tests.test_modern_hmc_pipeline import setup
    from bayesfilter.inference import tune_hmc_kernel
    binding, search = setup()
    result = tune_hmc_kernel(adapter=binding._base_adapter, initial_position=binding.initial_active_state,
                             config=search, candidate_set_adapter=binding.typed_adapter).result
    return result, binding


def test_native_audit_reconciles_all_fresh_trials(real_bridge_evidence):
    from bayesfilter.testing.inference_validation.modern_bridge_audit import audit_native_evidence
    result, binding = real_bridge_evidence
    audit = audit_native_evidence(result, binding)
    assert audit['returned_chunk_count'] == 64
    assert audit['attempted_chunk_count'] == 64
    assert audit['unobserved_attempted_chunk_count'] == 0
    assert audit['charged_gradient_work'] == audit['expected_completed_gradient_work']
    assert len({tuple(row['seed']) for row in audit['native_call_audit']}) == 64


@pytest.mark.parametrize('defect', ['calls', 'seed', 'profile', 'dropped_trial', 'uncharged_trial'])
def test_native_audit_rejects_corrupted_observations(real_bridge_evidence, defect):
    from copy import deepcopy
    from dataclasses import replace
    from bayesfilter.testing.inference_validation.modern_bridge_audit import audit_native_evidence
    result, original = real_bridge_evidence
    binding = SimpleNamespace(config=original.config, _spec=original._spec, _runtime=original._runtime,
        _evidence=deepcopy(original._evidence), _partial={}, work_seed=original.work_seed)
    evidence = next(iter(binding._evidence.values()))
    chunk = evidence['chunks'][0]
    if defect == 'calls':
        chunk['runtime']['value_score_row_evaluations'] += 1
    elif defect == 'seed':
        chunk['seed'][0] += 1
    elif defect == 'profile':
        chunk['runtime']['transition_spec']['profile']['family'] = 'mams'
    elif defect == 'dropped_trial':
        evidence['chunks'].pop()
    else:
        events = list(result.accounting_events)
        events.pop(next(i for i, e in enumerate(events) if e['event'] == 'numerical_chunk_charged'))
        result = replace(result, accounting_events=tuple(events))
    with pytest.raises(ValueError):
        audit_native_evidence(result, binding)


def test_failed_native_attempt_is_charged_without_fake_returned_evidence(real_bridge_evidence):
    from copy import deepcopy
    from dataclasses import replace
    from bayesfilter.testing.inference_validation.modern_bridge_audit import audit_native_evidence
    result, original = real_bridge_evidence
    evidence = deepcopy(original._evidence)
    digest, last = list(evidence.items())[-1]
    evidence.pop(digest)
    work_id = last['work']['work_item_id']
    binding = SimpleNamespace(config=original.config, _spec=original._spec, _runtime=original._runtime,
        _evidence=evidence, _partial={work_id: last['chunks'][:-1]}, work_seed=original.work_seed)
    result = replace(result, work_items=tuple(replace(w, status='interrupted') if w.work_item_id == work_id else w
                                             for w in result.work_items))
    audit = audit_native_evidence(result, binding)
    assert audit['attempted_chunk_count'] == 64
    assert audit['returned_chunk_count'] == 63
    assert audit['unobserved_attempted_chunk_count'] == 1


@pytest.fixture(scope='module')
def early_veto_bridge_evidence():
    """Actual modern kernel trials with a deterministic score-health veto."""
    from dataclasses import replace
    import tensorflow as tf
    from tests.test_modern_hmc_pipeline import setup
    from bayesfilter.inference import tune_hmc_kernel
    binding, search = setup()
    original = binding._run

    def poisoned(candidate, state, count, seed):
        result = original(candidate, state, count, seed)
        return replace(result, trace={**result.trace,
                                      'target_score_finite': tf.zeros([count, 4], tf.bool)})

    binding._run = poisoned
    try:
        result = tune_hmc_kernel(adapter=binding._base_adapter,
            initial_position=binding.initial_active_state, config=search,
            candidate_set_adapter=binding.typed_adapter).result
    finally:
        binding._run = original
    return result, binding


def test_native_audit_accepts_valid_early_candidate_veto(early_veto_bridge_evidence):
    from bayesfilter.testing.inference_validation.modern_bridge_audit import audit_native_evidence
    result, binding = early_veto_bridge_evidence
    assert result.completion_status == 'complete'
    assert not result.verified_candidate_ids
    assert len(binding._evidence) == 2
    for row in binding._evidence.values():
        assert row['hard_veto_stop']['completed_trials'] == 1
        assert row['hard_veto_stop']['unstarted_trials'] == 15
    audit = audit_native_evidence(result, binding)
    assert audit['attempted_chunk_count'] == audit['returned_chunk_count'] == 2
    assert audit['unobserved_attempted_chunk_count'] == 0
    assert audit['charged_gradient_work'] == audit['expected_completed_gradient_work']


@pytest.mark.parametrize('defect', ['inventory', 'cost', 'health', 'schema', 'analysis'])
def test_native_audit_rejects_corrupted_early_veto(early_veto_bridge_evidence, defect):
    from copy import copy, deepcopy
    from bayesfilter.testing.inference_validation.modern_bridge_audit import audit_native_evidence
    result, original = early_veto_bridge_evidence
    binding = copy(original)
    binding._evidence = deepcopy(original._evidence)
    binding._analysis_cache = {}
    numerical = next(iter(binding._evidence.values()))
    if defect == 'inventory':
        numerical['hard_veto_stop']['unstarted_trials'] = 0
    elif defect == 'cost':
        numerical['cost']['gradient_work'] += 1
    elif defect == 'health':
        numerical['trials'][0]['health']['hard_vetoes'] = []
    elif defect == 'schema':
        numerical['schema'] = 'bayesfilter.hmc_replicated_numerical_evidence.v2'
    else:
        numerical['analysis']['promotion_eligible'] = True
    with pytest.raises(ValueError):
        audit_native_evidence(result, binding)
