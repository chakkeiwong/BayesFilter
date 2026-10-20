"""CPU-only control tests for bounded transport calibration; no accuracy claim."""
import importlib
import sys
from pathlib import Path
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'docs/benchmarks'))
campaign=importlib.import_module('run_sqmc_expanded_comparison')
repair=importlib.import_module('run_sqmc_expanded_repair')


def test_repair_profile_keeps_target_and_disjoint_partitions(monkeypatch):
    names=('SCOPES','CAL','VAL','CLAIM','FILTER','ENTRY_POINT','EXTRA_SOURCE_PATHS','EXTRA_CONTRACT','CONTROL_PREPARER')
    before=campaign.numerical_sources(campaign.source_hashes())
    for name in names:monkeypatch.setattr(campaign,name,getattr(campaign,name))
    repair.configure_profile()
    contract=campaign.comparison_contract()
    assert len(contract['scopes'])==6
    assert {s[2] for s in contract['scopes']}=={3,10}
    assert all(s[4]>=1000 for s in contract['scopes'])
    assert set(contract['calibration']).isdisjoint(contract['validation']+contract['final_data'])
    assert set(contract['validation']).isdisjoint(contract['final_data'])
    assert not {195001,195002,196001,197001,197002,198001,198002}.intersection(
        contract['calibration']+contract['validation']+contract['final_data']+contract['final_filter'])
    assert campaign.numerical_sources(campaign.source_hashes())==before


def test_nomination_rejects_valid_but_insufficient_mass_margin():
    output=(tf.constant(-3.,tf.float64),tf.constant([.1],tf.float64),
        dict(program_valid=tf.constant([True,True]),higher_moment_valid=tf.constant([True,True]),
             row_error=tf.constant([1e-15,1e-15],tf.float64),column_tv=tf.constant([2e-6,1e-8],tf.float64)))
    result=repair.summarize_trace(output)
    assert result['valid'] is True
    assert result['nomination_pass'] is False


def test_nomination_requires_every_time_step_to_be_valid():
    output=(tf.constant(-3.,tf.float64),tf.constant([.1],tf.float64),
        dict(program_valid=tf.constant([False,True]),higher_moment_valid=tf.constant([True,True]),
             row_error=tf.constant([1e-15,1e-15],tf.float64),column_tv=tf.constant([1e-9,1e-9],tf.float64)))
    assert repair.summarize_trace(output)['nomination_pass'] is False
