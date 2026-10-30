"""Fresh closeout reuses checked arithmetic without borrowing live tuning trust."""
from copy import deepcopy
import json

import pytest
import tensorflow as tf

from bayesfilter.inference import (
    export_hmc_candidate_retained_runners, load_hmc_candidate_retained_runners,
    load_numerical_tuning_checkpoint, tune_hmc_kernel, tune_fixed_transport_hmc_kernel,
)
from bayesfilter.inference import hmc_acceptance_trials as trials
from bayesfilter.inference.hmc_candidate_set_execution import (
    HMCCandidateExecutionBinding, _ISSUER, _REPLAY_ANALYSES,
    _fresh_numerical_replay_scope, _json_copy,
)
from bayesfilter.inference.hmc_candidate_set_tuning import _json_native_sha256
from tests.test_hmc_replicated_batch_integration import setup


def count_reconstructions(monkeypatch):
    calls = []
    original = trials._assemble_trials

    def counted(runtime, work, chunks, **kwargs):
        calls.append((id(runtime), work.work_item_id))
        return original(runtime, work, chunks, **kwargs)

    monkeypatch.setattr(trials, "_assemble_trials", counted)
    return calls


@pytest.mark.parametrize("route", ["ordinary", "fixed_transport"])
def test_two_cold_readers_equal_scoped_closeout_and_reconstruct_each_record_once(tmp_path, monkeypatch, route):
    target, binding, search = setup(batch_size=8, positive=True, route=route)
    kwargs = dict(initial_position=binding.initial_active_state, config=search,
                  candidate_set_adapter=binding.typed_adapter, output_dir=tmp_path / "tuning")
    run = (tune_hmc_kernel(adapter=target, **kwargs) if route == "ordinary" else
           tune_fixed_transport_hmc_kernel(base_adapter=target, fixed_transport=binding.fixed_transport, **kwargs))
    assert len(run.result.verified_candidate_ids) == 2
    paths = export_hmc_candidate_retained_runners(candidate_set_result=run.result,
        retained_binding=binding, output_dir=tmp_path / "members")
    checkpoint = tmp_path / "tuning/tuning_checkpoint.json"
    calls = count_reconstructions(monkeypatch)
    cold_members = load_hmc_candidate_retained_runners(paths.values(), adapter=target)
    cold_binding, cold_controller = load_numerical_tuning_checkpoint(checkpoint, adapter=target)
    assert len(calls) == 2 * len(binding._evidence)
    calls.clear()
    with _fresh_numerical_replay_scope():
        # Even the original live cache cannot seed this operation's memo.
        for evidence in binding._evidence.values():
            binding.evidence_analysis(evidence)
        assert not _REPLAY_ANALYSES.get()[1]
        members = load_hmc_candidate_retained_runners(paths.values(), adapter=target)
        restored, controller = load_numerical_tuning_checkpoint(checkpoint, adapter=target)
        assert len(calls) == len(binding._evidence)
        assert restored._evidence == cold_binding._evidence
        assert controller.result().candidate_states == cold_controller.result().candidate_states
        assert set(members) == set(paths) == set(run.result.verified_candidate_ids)
        for cid, member in members.items():
            assert member.member_hash == cold_members[cid].member_hash
            tf.debugging.assert_equal(member.initial_active_state, cold_members[cid].initial_active_state)
        with pytest.raises(ValueError, match="duplicate"):
            load_hmc_candidate_retained_runners([*paths.values(), next(iter(paths.values()))], adapter=target)
        from tests.test_hmc_candidate_set_execution import GaussianTarget
        with pytest.raises(ValueError):
            load_hmc_candidate_retained_runners(paths.values(), adapter=GaussianTarget(scale=2.))
        path = next(iter(paths.values()))
        original_member = json.loads(path.read_text())
        for damage in ("member_hash", "candidate_id", "bundle_hash"):
            damaged = deepcopy(original_member)
            if damage == "bundle_hash":
                damaged["evidence_bundle"]["content_hash"] = "wrong"
            else:
                damaged[damage] = "wrong"
            damaged.pop("content_hash")
            damaged["content_hash"] = _json_native_sha256(damaged)
            path.write_text(json.dumps(damaged))
            with pytest.raises(ValueError):
                load_hmc_candidate_retained_runners(paths.values(), adapter=target)
        path.write_text(json.dumps(original_member))
    assert _REPLAY_ANALYSES.get() is None


@pytest.fixture(scope="module")
def small_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("scoped-replay")
    target, binding, search = setup(batch_size=2)
    tune_hmc_kernel(adapter=target, initial_position=binding.initial_active_state,
        config=search, candidate_set_adapter=binding.typed_adapter, output_dir=root)
    assert binding._evidence
    return target, binding, root / "tuning_checkpoint.json"


def clone_binding(binding):
    clone = HMCCandidateExecutionBinding(_ISSUER, adapter=binding._base_adapter, spec=deepcopy(binding._spec))
    clone._replicated_evidence_rungs = binding._replicated_evidence_rungs
    clone._evidence = deepcopy(binding._evidence)
    return clone


def test_old_binding_cold_recomputation_cannot_seed_new_reader(small_checkpoint, monkeypatch):
    _, binding, _ = small_checkpoint
    old = clone_binding(binding)
    row = next(iter(binding._evidence.values()))
    calls = count_reconstructions(monkeypatch)
    with _fresh_numerical_replay_scope():
        old.evidence_analysis(row)
        assert not _REPLAY_ANALYSES.get()[1]
        fresh = clone_binding(binding)
        fresh.evidence_analysis(row)
        assert len(calls) == 2


def test_nested_exception_and_later_scopes_start_cold(small_checkpoint, monkeypatch):
    _, binding, checkpoint = small_checkpoint
    row = next(iter(binding._evidence.values()))
    calls = count_reconstructions(monkeypatch)
    with _fresh_numerical_replay_scope():
        clone_binding(binding).evidence_analysis(row)
        outer = _REPLAY_ANALYSES.get()
        with pytest.raises(RuntimeError, match="reader failed"):
            with _fresh_numerical_replay_scope():
                clone_binding(binding).evidence_analysis(row)
                raise RuntimeError("reader failed")
        assert _REPLAY_ANALYSES.get() is outer
        clone_binding(binding).evidence_analysis(row)
        assert len(calls) == 2
    with _fresh_numerical_replay_scope():
        clone_binding(binding).evidence_analysis(row)
    assert len(calls) == 3
    load_numerical_tuning_checkpoint(checkpoint, adapter=binding._base_adapter)
    assert len(calls) == 3 + len(binding._evidence)


@pytest.mark.parametrize("damage", ["binding", "rungs", "seed", "sample", "trial_score"])
def test_changed_reconstruction_inputs_cannot_use_shared_analysis(small_checkpoint, monkeypatch, damage):
    _, binding, _ = small_checkpoint
    row = next(iter(binding._evidence.values()))
    calls = count_reconstructions(monkeypatch)
    with _fresh_numerical_replay_scope():
        expected = _json_copy(clone_binding(binding).evidence_analysis(row))
        other = clone_binding(binding)
        changed = deepcopy(row)
        if damage == "binding":
            other.binding_hash = "another-binding"
        elif damage == "rungs":
            other._replicated_evidence_rungs = (1, 4)
        elif damage == "seed":
            changed["chunks"][0]["seed"][0] ^= 1
        elif damage == "sample":
            changed["chunks"][0]["samples"]["sha256"] = "bad-checksum"
        else:
            changed["trials"][0]["scores"]["start_scores"][0] = .123
        with pytest.raises(ValueError):
            other.evidence_analysis(changed)
        # Returned values are independent copies, including nested health data.
        result = clone_binding(binding).evidence_analysis(row)
        result["acceptance_evidence"]["completed_trials"] = -10
        assert _json_copy(clone_binding(binding).evidence_analysis(row)) == expected


@pytest.mark.parametrize("damage", ["checksum", "rehashed_score", "missing_attempt"])
def test_checkpoint_file_checks_remain_after_prior_complete_replay(small_checkpoint, tmp_path, damage):
    import shutil
    target, _, original = small_checkpoint
    copied = tmp_path / "tuning"
    shutil.copytree(original.parent, copied)
    checkpoint = copied / original.name
    with _fresh_numerical_replay_scope():
        load_numerical_tuning_checkpoint(checkpoint, adapter=target)
        payload = json.loads(checkpoint.read_text())
        if damage == "missing_attempt":
            events = payload["result"]["accounting_events"]
            index = next(i for i, row in enumerate(events) if row["event"] == "numerical_chunk_charged")
            del events[index]
            result = payload["result"]
            result.pop("result_hash")
            result["result_hash"] = _json_native_sha256(result)
        else:
            old_hash = payload["numerical_evidence_hashes"][0]
            path = copied / "numerical_evidence" / (old_hash + ".json")
            evidence = json.loads(path.read_text())
            evidence["trials"][0]["scores"]["start_scores"][0] = .123
            if damage == "rehashed_score":
                new_hash = _json_native_sha256(evidence)
                path = path.with_name(new_hash + ".json")
                payload["numerical_evidence_hashes"][0] = new_hash
            path.write_text(json.dumps(evidence))
        payload.pop("content_hash")
        payload["content_hash"] = _json_native_sha256(payload)
        checkpoint.write_text(json.dumps(payload))
        with pytest.raises(ValueError):
            load_numerical_tuning_checkpoint(checkpoint, adapter=target)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "changed"])
def test_shared_reconstruction_still_checks_each_readers_predecessor_inventory(small_checkpoint, damage):
    _, binding, _ = small_checkpoint
    row = next(row for row in binding._evidence.values() if row["work"]["predecessor_work_id"] is not None)
    prior_hash = row["predecessor_evidence_hash"]
    with _fresh_numerical_replay_scope():
        clone_binding(binding).evidence_analysis(row)
        other = clone_binding(binding)
        if damage == "missing":
            del other._evidence[prior_hash]
        elif damage == "duplicate":
            other._evidence["duplicate"] = deepcopy(other._evidence[prior_hash])
        else:
            other._evidence[prior_hash]["analysis"]["acceptance"] = .123
        with pytest.raises(ValueError, match="predecessor"):
            other.evidence_analysis(row)
