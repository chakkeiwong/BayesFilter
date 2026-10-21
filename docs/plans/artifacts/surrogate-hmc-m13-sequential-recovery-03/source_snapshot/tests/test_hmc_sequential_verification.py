"""Deterministic control-flow regression; no inference evidence."""
from types import SimpleNamespace
from collections import namedtuple
import copy
import pytest
import tensorflow as tf
from bayesfilter.inference.hmc_sequential_verification import run_sequential_verification

Chunk = namedtuple("Chunk", "states final_state")

def exercise(rhats, *, fail_at=None, permit=None):
    # Deliberately small mechanics-only counts, not an admissible public config.
    cfg = SimpleNamespace(warmup_results=8, warmup_max_results=16,
        verification_results=4, verification_max_results=12, chunk_results=2,
        check_every_results=4, warmup_check_window_results=4, warmup_rhat_max=1.05,
        fresh_seed=(72, 31), nomination=SimpleNamespace(parameter_dimension=1))
    inspected, saved, updates, starts = [], [], [], []
    values = iter(rhats)
    def run(start, seed, count):
        starts.append(start)
        draws = start[None, ...] + tf.cast(tf.range(1,count+1)[:,None,None], tf.float64)
        return Chunk(draws, draws[-1])
    def diagnose(samples):
        inspected.append(samples)
        x = tf.constant([next(values)], tf.float64)
        return dict(maximum=x, bulk=x, folded=x)
    def archive(phase, index, chunk, initial, seed):
        saved.append((phase, chunk, initial, seed))
        return dict(path=str(len(saved)))
    result = run_sequential_verification(config=cfg, initial_state=tf.zeros([4,1],tf.float64),
        run_chunk=run, health=lambda chunk,start: dict(healthy=len(saved)!=fail_at),
        diagnose=diagnose, transform=lambda x:x, archive=archive,
        persist=lambda p:updates.append(copy.deepcopy(p)), verification_threshold=1.01,
        permit_chunk=permit)
    return result, inspected, saved, updates

def test_failed_first_looks_extend_without_reset_and_use_correct_windows():
    result, checked, saved, updates = exercise([1.08, 1.02, 1.04, 1.005])
    assert result["passed"] and result["counts"] == dict(warmup=12, verification=8)
    # Warm-up discards older diagnostics; fresh verification keeps all draws.
    assert [len(x) for x in checked] == [4,4,4,8]
    assert float(checked[0][0,0,0]) == 5.
    assert float(checked[1][0,0,0]) == 9.
    assert float(checked[2][0,0,0]) == float(checked[3][0,0,0]) == 13.
    for previous, following in zip(saved,saved[1:]):
        tf.debugging.assert_equal(previous[1].final_state, following[2])
    assert len(set(tuple(x) for x in result["used_seeds"])) == len(saved)
    assert updates[-1]["next_phase_plan"]["phase"] == "retained_assessment"
    assert updates[-1]["next_phase_plan"]["required_criterion"] == "retained_rhat_ess_and_reference"
    assert "maximum_per_chain" not in updates[-1]["next_phase_plan"]

def test_warmup_cap_never_starts_verification():
    result, _, saved, _ = exercise([1.1,1.1,1.1])
    assert result["status"] == "WARMUP_MAXIMUM_INCONCLUSIVE"
    assert result["counts"] == dict(warmup=16,verification=0)
    assert all(x[0]=="warmup" for x in saved)

def test_verification_cap_never_promotes():
    result, checked, _, _ = exercise([1.01,1.02,1.02,1.02])
    assert result["status"] == "VERIFICATION_MAXIMUM_INCONCLUSIVE"
    assert [len(x) for x in checked] == [4,4,8,12]
    assert not result["passed"]

def test_health_failure_archived_and_stops_before_rhat():
    result, checked, saved, updates = exercise([], fail_at=1)
    assert result["status"] == "HEALTH_VETO" and not checked
    assert len(saved)==2 and updates[-1]["next_phase_plan"]["continuation_veto"]

def test_budget_exhaustion_before_first_chunk_has_terminal_receipt():
    result, _, saved, updates = exercise([], permit=lambda phase,n:False)
    assert result["status"] == "BUDGET_EXHAUSTED" and not saved
    assert updates[-1]["status"] == "BUDGET_EXHAUSTED"

def test_bad_public_policy_cannot_be_loaded():
    from bayesfilter.testing.surrogate_hmc_m13_public_tuning import tuning_config
    from dataclasses import replace
    for changes in (dict(warmup_results=500), dict(warmup_max_results=11000),
                    dict(verification_max_results=500), dict(chunk_results=300),
                    dict(check_every_results=500), dict(warmup_rhat_max=1.1)):
        with pytest.raises(ValueError):
            replace(tuning_config("policy-test"), **changes)
