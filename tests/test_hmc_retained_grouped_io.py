"""Grouped replay checks the entire scope and preserves every member identity."""
from copy import deepcopy
import json

import pytest
import tensorflow as tf

from bayesfilter.inference import (
    export_hmc_candidate_retained_runners, load_hmc_candidate_retained_runners,
    load_hmc_candidate_retained_runner, tune_hmc_kernel,
)
from bayesfilter.inference import hmc_candidate_set_retained as retained
from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCControllerConfig, HMCTuningCandidateSetController, _sha256, _json_native_sha256,
)
from tests.test_hmc_candidate_set_execution import GaussianTarget, execution_config, make_binding


@pytest.fixture(scope='module')
def tuned_group():
    binding = make_binding(config=execution_config(chain_mode='batched', use_xla=True,
        non_xla_reason=None, reuse_leapfrog_graphs=True))
    search = HMCControllerConfig(primary_l_grid=(2, 3),
        epsilon_by_l=((2, (1.1, 1.3, 1.5)), (3, (1.1, 1.3, 1.5))),
        total_budget_units=40, repair_reserve_units=3)
    run = tune_hmc_kernel(adapter=binding._base_adapter, initial_position=binding.initial_active_state,
        config=search, candidate_set_adapter=binding.typed_adapter)
    assert len(run.result.verified_candidate_ids) >= 2
    return binding, run.result


def test_group_checks_common_evidence_once_and_matches_individual_replay(tuned_group, tmp_path, monkeypatch):
    binding, result = tuned_group
    checks = []
    original = retained._validate_member_set
    def counted(*args, **kwargs):
        checks.append(1)
        return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(retained, '_validate_member_set', counted)
        paths = export_hmc_candidate_retained_runners(candidate_set_result=result,
            retained_binding=binding, output_dir=tmp_path)
        assert len(checks) == 1
        grouped = load_hmc_candidate_retained_runners(paths.values(), adapter=GaussianTarget())
        assert len(checks) == 2
    assert set(grouped) == set(paths) == set(result.verified_candidate_ids)
    assert len(list(tmp_path.glob('evidence-*.json'))) == 1
    assert len({id(r._binding) for r in grouped.values()}) == 1
    for cid, path in paths.items():
        individual = load_hmc_candidate_retained_runner(path, adapter=GaussianTarget())
        assert individual.member_hash == grouped[cid].member_hash
        tf.debugging.assert_equal(individual.initial_active_state, grouped[cid].initial_active_state)


@pytest.mark.parametrize('damage', ['endpoint', 'member_hash', 'bundle_hash', 'unverified_id'])
def test_group_rejects_rehashed_member_damage(tuned_group, tmp_path, damage):
    binding, result = tuned_group
    paths = export_hmc_candidate_retained_runners(candidate_set_result=result,
        retained_binding=binding, output_dir=tmp_path)
    path = list(paths.values())[-1]
    payload = json.loads(path.read_text())
    if damage == 'endpoint':
        payload['verified_endpoint'] = retained._tensor_payload(tf.zeros([4, 2], tf.float64))
    elif damage == 'member_hash':
        payload['member_hash'] = '0' * 64
    elif damage == 'bundle_hash':
        payload['evidence_bundle']['content_hash'] = '0' * 64
    else:
        payload['candidate_id'] = 'not-a-verified-candidate'
    payload.pop('content_hash')
    payload['content_hash'] = _json_native_sha256(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_hmc_candidate_retained_runners(paths.values(), adapter=GaussianTarget())


def test_group_revalidates_later_live_evidence_mutation(tuned_group, tmp_path):
    binding, result = tuned_group
    paths = export_hmc_candidate_retained_runners(candidate_set_result=result,
        retained_binding=binding, output_dir=tmp_path)
    loaded = load_hmc_candidate_retained_runners(paths.values(), adapter=GaussianTarget())
    runner = list(loaded.values())[-1]
    first = next(iter(runner._binding._evidence.values()))
    first['analysis']['acceptance'] = .123
    with pytest.raises(ValueError, match='corrupt numerical evidence'):
        runner.export(tmp_path/'after-mutation.json')


def test_shared_invalidity_revokes_previously_verified_group(tuned_group, tmp_path):
    binding, result = tuned_group
    controller = HMCTuningCandidateSetController.from_result_payload(
        retained._result_payload(result))
    controller._scope_invalid = True
    with pytest.raises(ValueError, match='shared-invalid'):
        export_hmc_candidate_retained_runners(candidate_set_result=controller.result(),
            retained_binding=binding, output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_portable_members_remain_compatible_and_duplicates_are_rejected(tuned_group, tmp_path):
    binding, result = tuned_group
    cid = result.verified_candidate_ids[0]
    original = retained.build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=result, candidate_id=cid, retained_binding=binding)
    path = original.export(tmp_path/'portable.json', portable=True)
    loaded = load_hmc_candidate_retained_runners([path], adapter=GaussianTarget())
    assert loaded[cid].member_hash == original.member_hash
    with pytest.raises(ValueError, match='duplicate'):
        load_hmc_candidate_retained_runners([path, path], adapter=GaussianTarget())
    with pytest.raises(TypeError, match='sequence'):
        load_hmc_candidate_retained_runners(path, adapter=GaussianTarget())


def test_native_json_hash_preserves_nested_wire_identity_and_rejects_unsupported_values():
    record = {'trace': ['unicode \u03b5', None, True, False, -0., 1.e-50, (1, 2.5)],
              'nested': {'z': 'binary-as-base64', 'a': [{'x': -3}]}}
    assert _json_native_sha256(record) == _sha256(record)
    assert _json_native_sha256(json.loads(json.dumps(record))) == _sha256(record)
    for bad in (float('nan'), float('inf'), object()):
        damaged = deepcopy(record)
        damaged['bad'] = bad
        with pytest.raises((ValueError, TypeError)):
            _json_native_sha256(damaged)

