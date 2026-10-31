"""C3 continuation accounting and status updates; no scientific inference."""
import importlib.util
from pathlib import Path
import pytest

def load(name):
    path=Path(__file__).resolve().parents[1]/'scripts/diagnostics'/name
    spec=importlib.util.spec_from_file_location(name[:-3],path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

observer=load('continue_modern_hmc_c3.py')
driver=load('run_modern_hmc_macrofinance_posterior.py')

def example():
    prep=dict(prior_upper_balance_seconds=10000.,reference_charge_seconds=300.,
        preparation_reserve_seconds=1000.,separate_sir_seconds=7363.)
    settlement=dict(status='completed',inclusive_charge_seconds=170.)
    price=dict(status='completed',stage='price',elapsed_seconds=170.,rows=[
        dict(L=L,epsilon=e,status='priced',seconds=10.,transitions_per_chain=1000) for L,e in driver.PAIRS])
    return prep,settlement,price

def test_allocation_charges_actual_once_and_leaves_sir_untouched():
    prep,settlement,price=example()
    result=observer.allocation(prep,settlement,price,driver.posterior_quote)
    assert result['balance_before_posterior_seconds']==8530
    assert result['posterior_reservation_seconds']==2500
    assert result['unreserved_upper_balance_seconds']==6030
    assert result['separate_sir_seconds']==7363

def test_insufficient_budget_and_failed_member_cannot_launch():
    prep,settlement,price=example();prep['prior_upper_balance_seconds']=2000.
    with pytest.raises(ValueError,match='affordable'):
        observer.allocation(prep,settlement,price,driver.posterior_quote)
    prep,settlement,price=example();price['rows'][2]['status']='numerical_health_failed'
    with pytest.raises(ValueError,match='seven'):
        observer.allocation(prep,settlement,price,driver.posterior_quote)
    settlement['status']='error'
    with pytest.raises(ValueError,match='complete'):
        observer.allocation(prep,settlement,price,driver.posterior_quote)

def test_program_refresh_preserves_existing_text_and_replaces_only_own_block(tmp_path):
    folder=tmp_path/'docs/plans';folder.mkdir(parents=True)
    names=('bayesfilter-modern-hmc-implementation-plan-2026-10-07.md',
           'bayesfilter-modern-hmc-active-checkpoint-2026-10-13.md')
    for name in names:(folder/name).write_text('# Existing master'+chr(10)+'Keep independent work.'+chr(10))
    record=dict(status='waiting_for_prices',updated_utc='2026-10-29T15:00Z')
    observer.refresh_program(tmp_path,record)
    record.update(status='posterior_running',updated_utc='2026-10-29T16:00Z')
    observer.refresh_program(tmp_path,record)
    for name in names:
        text=(folder/name).read_text()
        assert text.count(observer.MARKER)==2
        assert 'Keep independent work.' in text
        assert 'waiting_for_prices' not in text
        assert 'posterior_running' in text
