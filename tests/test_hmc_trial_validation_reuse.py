"""Call-local seed reuse preserves every row/charge check.

The tensors below are codec fixtures, not simulated HMC or admission evidence.
Actual public routes and multi-chunk replay are covered by the integration suite.
"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference import hmc_acceptance_trials as trials
from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
from bayesfilter.inference.hmc_candidate_set_execution import _tensor_payload
from bayesfilter.inference.hmc_candidate_set_tuning import HMCWorkItem


def reference_seed(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).digest()
    return tuple(int.from_bytes(digest[i:i+4], "big") & 0x7fffffff for i in (0, 4))


def fixture(batch):
    policy = HMCReplicatedAcceptancePolicy(trial_num_results=65, discarded_prefix=3,
        base_repetitions=batch, max_repetitions=batch, max_candidates=1,
        search_family_alpha=.05, verification_family_alpha=.05,
        diagnostic_family_alpha=.05, temporal_tolerance=.1)
    work = HMCWorkItem("work", "candidate", "record", "family", "measurement",
        "cohort", 0, 0, trial_range=(0, batch))
    runtime = SimpleNamespace(
        config=SimpleNamespace(seed=(20261003, 2888), acceptance_policy=policy,
            chunk_max_results=68, replicated_trial_batch_size=batch),
        scope=SimpleNamespace(payload=lambda: {"scope": "codec-fixture"}),
        initial_active_state=tf.zeros([4, 2], tf.float64), binding_hash="binding",
        _replicated_evidence_rungs=(1,), _partial={}, _evidence={},
        trial_seed_lineage=lambda seed: (tuple(seed),))
    root = reference_seed({"seed": runtime.config.seed, "scope": runtime.scope.payload(),
        "candidate": work.candidate_record_hash, "stage": work.stage,
        "protocol": policy.identity})
    seeds = [trials.chunk_seed(reference_seed({"stage_seed": root, "trial_ordinal": i}), 0)
             for i in range(batch)]
    samples = _tensor_payload(tf.ones([68, 4, 2], tf.float64))
    chunks = [dict(trial_ordinal=i, trial_chunk_index=0, count=68, seed=list(seed),
        initial_state=_tensor_payload(runtime.initial_active_state), binding_hash="binding",
        work=json.loads(json.dumps(work.payload())), samples=deepcopy(samples), trace={},
        runtime=dict(trial_batch_size=batch, trial_batch_row=i,
            seed_layout="independent_original_tfp_sample_chain_streams_v1",
            trial_batch_seeds=[list(s) for s in seeds])) for i, seed in enumerate(seeds)]
    runtime._partial = {work.work_item_id: chunks}
    events = [dict(event="numerical_chunk_charged", work_item_id=work.work_item_id,
        chunk_index=i, trial_ordinal=i, trial_chunk_index=0, seed=list(seed),
        transitions=68*4, gradient_work=68*4*3) for i, seed in enumerate(seeds)]
    result = dict(work_items=[work.payload()], candidates=[dict(candidate_id="candidate", leapfrog_steps=2)],
        accounting_events=events)
    return runtime, work, chunks, result


@pytest.mark.parametrize("batch", [1, 8, 32])
def test_one_work_root_per_call_preserves_reference_seeds_and_reordered_charges(batch, monkeypatch):
    runtime, work, chunks, result = fixture(batch)
    original = trials.work_seed
    calls = []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(trials, "work_seed", counted)
    trials.validate_chunks(runtime, work, chunks, complete=True)
    assert len(calls) == 1
    calls.clear()
    trials.initialize_seed_registry(runtime, result)
    expected = dict(runtime._trial_seed_registry)
    assert len(calls) == 1
    assert len(expected) == batch
    result["accounting_events"].reverse()
    result["accounting_events"].append(deepcopy(result["accounting_events"][0]))
    trials.initialize_seed_registry(runtime, result)
    assert runtime._trial_seed_registry == expected


@pytest.mark.parametrize("batch", [8, 32])
@pytest.mark.parametrize("row", [0, -1])
def test_every_row_metadata_is_checked_again_on_later_validation(batch, row):
    runtime, work, chunks, _ = fixture(batch)
    trials.validate_chunks(runtime, work, chunks, complete=True)
    chunks[row]["runtime"]["trial_batch_seeds"][-1][0] ^= 1
    with pytest.raises(ValueError, match="batch streams"):
        trials.validate_chunks(runtime, work, chunks, complete=True)


@pytest.mark.parametrize("batch", [8, 32])
def test_later_missing_charge_for_unsaved_last_row_cannot_reuse_prior_validation(batch):
    runtime, _, chunks, result = fixture(batch)
    trials.initialize_seed_registry(runtime, result)
    # One durable returned row still proves that the whole native batch started.
    del chunks[1:]
    result["accounting_events"].pop()
    with pytest.raises(ValueError, match="batch has an uncharged row"):
        trials.initialize_seed_registry(runtime, result)


@pytest.mark.parametrize("field", [
    "schema", "evidence_unit", "trial_range", "predecessor_work_id",
    "work_item_id", "candidate_id", "candidate_record_hash", "candidate_family_id",
    "stage", "cohort_id", "ordinal", "priority", "repair_action_id",
    "verification_attempt_id", "reservation_units", "evidence_rung", "evidence_multiplier",
])
def test_chunk_cannot_omit_any_issued_work_identity_field(field):
    runtime, work, chunks, _ = fixture(2)
    del chunks[-1]["work"][field]
    with pytest.raises(ValueError, match="trial chunk work mismatch"):
        trials.validate_chunks(runtime, work, chunks, complete=True)


@pytest.mark.parametrize("damage", ["empty", "extra_null"])
def test_chunk_work_must_match_the_complete_issued_record(damage):
    runtime, work, chunks, _ = fixture(2)
    if damage == "empty":
        chunks[-1]["work"] = {}
    else:
        chunks[-1]["work"]["unexpected"] = None
    with pytest.raises(ValueError, match="trial chunk work mismatch"):
        trials.validate_chunks(runtime, work, chunks, complete=True)


def test_chunk_work_status_can_advance_without_changing_its_identity():
    runtime, work, chunks, _ = fixture(2)
    assert isinstance(chunks[-1]["work"]["trial_range"], list)
    chunks[-1]["work"]["status"] = "completed"
    trials.validate_chunks(runtime, work, chunks, complete=True)
