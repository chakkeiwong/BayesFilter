"""Charges follow native chunk extents; retries cost work, not information."""
import copy
from dataclasses import replace
import hashlib
import json

import pytest

from bayesfilter.inference.hmc_candidate_set_adapters import run_typed_hmc_candidate_set
from bayesfilter.inference.hmc_acceptance_trials import initialize_seed_registry
from bayesfilter.testing.acceptance_decision_models import evidence_accounting
from tests.test_hmc_acceptance_trials import _replicated_setup


@pytest.fixture(scope="module")
def completed():
    binding, search = _replicated_setup()
    run = run_typed_hmc_candidate_set(binding.typed_adapter, search)
    return binding, run.result


def test_cumulative_rungs_count_unique_trials_and_every_native_chunk(completed):
    binding, result = completed
    report = evidence_accounting(binding, result)
    assert report["unique_complete_trials"] == report["valid_complete_trials"] == 2
    assert report["charged_chunks"] == 4
    assert report["attempted_transitions"] == report["completed_trial_transitions"] == 2*68*4
    assert report["gradient_work"] == 2*68*4*3
    assert report["attempted_work_outside_complete_trials"] == 0


def test_compact_health_observations_keep_raw_evidence_and_historical_readback(completed):
    from bayesfilter.inference.hmc_acceptance_trials import _analyze_trials
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCWorkItem
    binding,_ = completed
    raw = copy.deepcopy(next(iter(binding._evidence.values())))
    work = HMCWorkItem.from_payload(raw['work'])
    compact = _analyze_trials(binding,work,raw['trials'],raw['evidence_rungs'])
    historical = _analyze_trials(binding,work,raw['trials'],raw['evidence_rungs'],compact_health=False)
    assert compact['acceptance_evidence']['schema'].endswith('.v2')
    assert historical['acceptance_evidence']['schema'].endswith('.v1')
    for key in compact:
        if key != 'acceptance_evidence':
            assert compact[key] == historical[key]
    assert compact['acceptance_evidence']['statistics'] == historical['acceptance_evidence']['statistics']
    summary = compact['acceptance_evidence']['health_summary']
    assert summary['trial_count'] == len(raw['trials'])
    assert sum(summary['evidence_validity_counts'].values()) == len(raw['trials'])
    assert historical['acceptance_evidence']['health'] == [t['health'] for t in raw['trials']]
    # The compatibility path re-derives old v1 details from the original traces.
    raw['analysis'] = historical
    assert json.loads(json.dumps(binding.evidence_analysis(raw))) == json.loads(json.dumps(historical))
    raw['analysis']['acceptance_evidence']['schema'] = 'unknown-health-summary'
    with pytest.raises(ValueError,match='summary'):
        binding.evidence_analysis(raw)


def test_failed_call_is_charged_again_but_supplies_only_one_score(completed):
    binding, result = completed
    event = next(e for e in result.accounting_events if e["event"] == "numerical_chunk_charged")
    # Same stream is permitted for recovery of a lost call. There is no extra
    # independent score and the additional attempted call must not disappear.
    retried = replace(result, accounting_events=(*result.accounting_events, event))
    report = evidence_accounting(binding, retried)
    assert report["unique_complete_trials"] == 2
    assert report["charged_chunks"] == 5
    assert report["attempted_work_outside_complete_trials"] == 34*4*3


@pytest.mark.parametrize("mutation", ["undercharge", "wrong_gradient", "missing_charge", "wrong_seed"])
def test_plausible_aggregate_totals_cannot_hide_wrong_chunk_accounting(completed, mutation):
    binding, result = completed
    events = copy.deepcopy(list(result.accounting_events))
    index = next(i for i,e in enumerate(events) if e["event"] == "numerical_chunk_charged")
    if mutation == "undercharge":
        events[index]["transitions"] //= 2
        events[index]["gradient_work"] //= 2
    elif mutation == "wrong_gradient":
        events[index]["gradient_work"] -= 1
    elif mutation == "wrong_seed":
        events[index]["seed"] = (17, 23)
    else:
        # Preserve total cost by substituting a different chunk's charge.
        events[index] = copy.deepcopy(next(e for e in reversed(events)
                                          if e["event"] == "numerical_chunk_charged"))
    with pytest.raises(ValueError, match="cost|stream|charged attempt"):
        initialize_seed_registry(binding, replace(result, accounting_events=tuple(events)))


@pytest.mark.parametrize("mutation", ["duplicate_trial", "changed_trial", "duplicate_chunk"])
def test_cumulative_evidence_cannot_duplicate_or_change_information(completed, mutation, monkeypatch):
    binding, result = completed
    rows = copy.deepcopy(binding._evidence)
    row = list(rows.values())[-1]
    if mutation == "duplicate_trial":
        row["trials"].append(copy.deepcopy(row["trials"][0]))
    elif mutation == "changed_trial":
        row["trials"][0]["scores"]["start_scores"][0] = .123
    else:
        row["trials"][-1]["chunks"].append(copy.deepcopy(row["trials"][-1]["chunks"][0]))
    monkeypatch.setattr(binding, "_evidence", rows)
    with pytest.raises(ValueError, match="duplicate|changed"):
        evidence_accounting(binding, result)


def test_checkpoint_rejects_undercharge_even_with_recomputed_checksums(tmp_path):
    from bayesfilter.inference.hmc_candidate_set_checkpoint import load_numerical_tuning_checkpoint
    from bayesfilter.inference.hmc_candidate_set_tuning import _sha256
    from bayesfilter.inference.hmc_candidate_set_artifacts import _canonical
    binding, search = _replicated_setup()
    run_typed_hmc_candidate_set(binding.typed_adapter, search, output_dir=tmp_path)
    path = tmp_path/"tuning_checkpoint.json"
    payload = json.loads(path.read_text())
    result = payload["result"]
    event = next(e for e in result["accounting_events"] if e["event"] == "numerical_chunk_charged")
    reduction = event["gradient_work"] // 2
    event["transitions"] //= 2
    event["gradient_work"] -= reduction
    result["search_state"]["gradient_work"] -= reduction
    result.pop("result_hash")
    result["result_hash"] = hashlib.sha256(_canonical(result)).hexdigest()
    payload.pop("content_hash")
    payload["content_hash"] = _sha256(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="cost differs from frozen chunk"):
        load_numerical_tuning_checkpoint(path, adapter=binding._base_adapter)
