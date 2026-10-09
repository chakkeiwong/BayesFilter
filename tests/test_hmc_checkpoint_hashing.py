"""Checkpoint specialization preserves identities and live mutation checks."""
import json

import pytest

from bayesfilter.inference.hmc_candidate_set_checkpoint import (
    _json_native_sha256, write_numerical_tuning_checkpoint)
from bayesfilter.inference.hmc_candidate_set_tuning import _sha256


@pytest.mark.parametrize("payload", [
    {"unicode": "θ", "number": -0.0, "integer": 2**64, "none": None},
    {"nested": ({"2": [True, False, 1.234e-77]},), "bytes": "AAECAw=="},
    {"trace": {"status": [0, 0, 1]}, "scientific": "not admissible"},
])
def test_json_native_hash_matches_general_identity(payload):
    normalized = json.loads(json.dumps(payload))
    assert _json_native_sha256(normalized) == _sha256(normalized)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), object()])
def test_non_json_or_nonfinite_native_record_is_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        _json_native_sha256({"value": value})


def test_actual_checkpoint_rechecks_mutation_after_cached_persistence(tmp_path):
    from bayesfilter.inference import run_typed_hmc_candidate_set, load_numerical_tuning_checkpoint
    from tests.test_hmc_candidate_set_execution import make_binding, GaussianTarget
    from tests.test_hmc_candidate_set_tuning import _config

    binding = make_binding()
    result = run_typed_hmc_candidate_set(binding.typed_adapter,
        _config(grid=(3,), epsilons=((3, (1.3,)),)), output_dir=tmp_path).result
    assert binding._persisted_files and binding._evidence
    restored, controller = load_numerical_tuning_checkpoint(tmp_path / "tuning_checkpoint.json",
        adapter=GaussianTarget())
    assert controller.result().verified_candidate_ids == result.verified_candidate_ids
    key = next(iter(binding._evidence))
    binding._evidence[key]["analysis"]["acceptance"] = .123
    with pytest.raises(ValueError, match="corrupt live numerical evidence"):
        write_numerical_tuning_checkpoint(binding, result, tmp_path)
    restored._spec["config"]["measurement_num_results"] += 1
    with pytest.raises(ValueError, match="corrupt execution specification"):
        write_numerical_tuning_checkpoint(restored, controller.result(), tmp_path)


def test_internal_chunk_save_preserves_full_boundary_validation(tmp_path):
    from bayesfilter.inference import run_typed_hmc_candidate_set, load_numerical_tuning_checkpoint
    from tests.test_hmc_acceptance_trials import _replicated_setup
    binding, search = _replicated_setup()
    result = run_typed_hmc_candidate_set(binding.typed_adapter,search,output_dir=tmp_path).result
    # The returned result can have a later elapsed-time field than the last
    # controller callback. Compare two serializers of the same explicit state.
    write_numerical_tuning_checkpoint(binding,result,tmp_path)
    original = json.loads((tmp_path/'tuning_checkpoint.json').read_text())
    write_numerical_tuning_checkpoint(binding,result,tmp_path,_incremental=True)
    assert json.loads((tmp_path/'tuning_checkpoint.json').read_text()) == original
    restored,controller = load_numerical_tuning_checkpoint(tmp_path/'tuning_checkpoint.json',
                                                         adapter=binding._base_adapter)
    assert controller.result().candidate_states == result.candidate_states
    assert restored._evidence == binding._evidence
    key = next(iter(binding._evidence))
    binding._evidence[key]['analysis']['acceptance'] = .123
    # The next stage/final boundary must still reject a live mutation even
    # after its immutable file has been reused in a within-work save.
    write_numerical_tuning_checkpoint(binding,result,tmp_path,_incremental=True)
    with pytest.raises(ValueError,match='corrupt live numerical evidence'):
        write_numerical_tuning_checkpoint(binding,result,tmp_path)


@pytest.mark.parametrize('mutation',['deleted','changed'])
def test_incremental_save_checks_changed_or_missing_immutable_file(tmp_path,mutation):
    from bayesfilter.inference import run_typed_hmc_candidate_set
    from tests.test_hmc_acceptance_trials import _replicated_setup
    binding,search = _replicated_setup()
    result = run_typed_hmc_candidate_set(binding.typed_adapter,search,output_dir=tmp_path).result
    key = next(iter(binding._evidence))
    path = tmp_path/'numerical_evidence'/(key+'.json')
    if mutation == 'deleted':
        path.unlink()
        write_numerical_tuning_checkpoint(binding,result,tmp_path,_incremental=True)
        assert json.loads(path.read_text()) == binding._evidence[key]
    else:
        path.write_text('{}')
        with pytest.raises(ValueError,match='immutable numerical evidence collision'):
            write_numerical_tuning_checkpoint(binding,result,tmp_path,_incremental=True)


def test_incremental_scope_is_restored_after_error_and_nested_scope():
    from types import SimpleNamespace
    from bayesfilter.inference.hmc_candidate_runtime import incremental_chunk_checkpoint
    runtime = SimpleNamespace()
    with pytest.raises(RuntimeError):
        with incremental_chunk_checkpoint(runtime):
            assert runtime._incremental_checkpoint is True
            with incremental_chunk_checkpoint(runtime):
                assert runtime._incremental_checkpoint is True
            assert runtime._incremental_checkpoint is True
            raise RuntimeError('injected native failure')
    assert runtime._incremental_checkpoint is False


def test_member_recomputes_each_record_once_per_call_and_never_across_calls(monkeypatch):
    from collections import Counter
    from bayesfilter.inference import run_typed_hmc_candidate_set, build_retained_bound_hmc_archive_runner_from_candidate_set_result
    from tests.test_hmc_candidate_set_execution import make_binding
    from tests.test_hmc_candidate_set_tuning import _config
    binding = make_binding()
    result = run_typed_hmc_candidate_set(binding.typed_adapter,
        _config(grid=(3,),epsilons=((3,(1.3,)),))).result
    assert result.verified_candidate_ids
    original = binding.evidence_analysis
    calls = Counter()
    def counted(row):
        calls[row['work']['work_item_id']] += 1
        return original(row)
    monkeypatch.setattr(binding,'evidence_analysis',counted)
    for count in (1,2):
        build_retained_bound_hmc_archive_runner_from_candidate_set_result(
            candidate_set_result=result,candidate_id=result.verified_candidate_ids[0],retained_binding=binding)
        assert len(calls) == len(binding._evidence)
        assert set(calls.values()) == {count}
    next(iter(binding._evidence.values()))['analysis']['acceptance'] = .123
    with pytest.raises(ValueError,match='corrupt numerical evidence'):
        build_retained_bound_hmc_archive_runner_from_candidate_set_result(
            candidate_set_result=result,candidate_id=result.verified_candidate_ids[0],retained_binding=binding)
