"""Generated evidence reuses its analysis without weakening later reconstruction."""
import copy

import pytest

from bayesfilter.inference import hmc_acceptance_trials as trials
from bayesfilter.inference.hmc_candidate_set_adapters import run_typed_hmc_candidate_set
from bayesfilter.inference.hmc_candidate_set_execution import _json_copy
from bayesfilter.inference.hmc_candidate_set_tuning import _json_native_sha256


def setup(kind):
    if kind == "chunked":
        from tests.test_hmc_acceptance_trials import _replicated_setup
        return _replicated_setup()
    from tests.test_hmc_replicated_batch_integration import setup as batch_setup
    _, binding, search = batch_setup(batch_size=2 if kind == "batch" else 1)
    return binding, search


@pytest.mark.parametrize("kind", ["single", "chunked", "batch"])
def test_emission_assembles_each_work_once_and_matches_forced_reconstruction(kind, monkeypatch):
    binding, search = setup(kind)
    original = trials._assemble_trials
    calls = []

    def counted(runtime, work, chunks, **kwargs):
        calls.append(work.work_item_id)
        return original(runtime, work, chunks, **kwargs)

    monkeypatch.setattr(trials, "_assemble_trials", counted)
    run_typed_hmc_candidate_set(binding.typed_adapter, search)
    evidence = list(binding._evidence.items())
    assert evidence
    assert len(calls) == len(evidence) == len(set(calls))
    assert all("execution_failure" not in row for _, row in evidence)
    for digest, row in evidence:
        assert digest in binding._analysis_cache
        assert binding.evidence_analysis(row) == row["analysis"]
        returned = binding.evidence_analysis(row)
        returned["acceptance"] = 123.0
        assert binding.evidence_analysis(row) == row["analysis"]
    assert len(calls) == len(evidence)
    # A cache-free reconstruction is the independent raw-evidence comparator.
    # A fresh binding also starts empty; the public reload tests cover that path.
    binding._analysis_cache.clear()
    before = len(calls)
    for _, row in evidence:
        assert _json_copy(binding.evidence_analysis(row)) == row["analysis"]
    assert len(calls) - before == len(evidence)


@pytest.mark.parametrize("damage", ["score", "seed", "work", "rungs", "sample"])
def test_emission_cache_cannot_hide_changed_raw_evidence(damage):
    binding, search = setup("chunked")
    run_typed_hmc_candidate_set(binding.typed_adapter, search)
    original = list(binding._evidence.values())[-1]
    assert _json_native_sha256(original) in binding._analysis_cache
    changed = copy.deepcopy(original)
    if damage == "score":
        changed["trials"][-1]["scores"]["start_scores"] = [.7] * 4
    elif damage == "seed":
        changed["chunks"][0]["seed"] = [177, 913]
    elif damage == "work":
        changed["chunks"][0]["work"]["candidate_record_hash"] = "wrong"
    elif damage == "rungs":
        changed["evidence_rungs"] = [1, 4]
    else:
        changed["chunks"][0]["samples"]["sha256"] = "wrong"
    assert _json_native_sha256(changed) != _json_native_sha256(original)
    with pytest.raises(ValueError):
        binding.evidence_analysis(changed)
    assert binding.evidence_analysis(original) == original["analysis"]


def test_new_native_row_metadata_must_validate_before_cache_or_observation(monkeypatch):
    binding, search = setup("batch")
    original = binding._run_replicated_batch

    def damaged(*args, **kwargs):
        outputs = original(*args, **kwargs)
        outputs[-1].metadata["trial_batch_seeds"] = [[0, 0]]
        return outputs

    monkeypatch.setattr(binding, "_run_replicated_batch", damaged)
    with pytest.raises(ValueError, match="trial batch streams"):
        run_typed_hmc_candidate_set(binding.typed_adapter, search)
    assert not binding._evidence
    assert not binding._analysis_cache
