"""Independent checks of the rare-path arithmetic reference."""
import math
import pytest
from bayesfilter.testing.zhao_cui_pp_tail_reference import checked_log_transition


def test_zero_dynamics_matches_gaussian():
    value,record=checked_log_transition([0,0,0,0,1.2,.5],[50,5],[51,7])
    assert value==pytest.approx(-math.log(2*math.pi)-2*math.log(2)-5/8,abs=1e-14)
    assert record['relative_difference']<1e-40


def test_captured_overflow_retains_zero_weight():
    theta=[.5458831878914765,.3141535260459123,.8540393803979973,.8229186458492244,(115.23901531520356-90)/20,(22.241336190936302-20)/10]
    value,record=checked_log_transition(theta,[8514.423609668846,273.26053310934327],[115.37645634678204,.33444388051182194])
    assert value==-math.inf
    assert record['log10_absolute_log_density']==pytest.approx(1.927837598132897e23,rel=1e-12)


def test_integrator_stage_is_explicit():
    args=([.6,.3,.5,.5,1.2,.5],[50,5],[55,6])
    full,_=checked_log_transition(*args,k4_fraction=1.)
    half,_=checked_log_transition(*args,k4_fraction=.5)
    assert abs(full-half)>1e-4


@pytest.mark.parametrize('bad', [float('nan'),float('inf')])
def test_nonfinite_input_rejects(bad):
    with pytest.raises(ValueError):checked_log_transition([.6,.3,.5,.5,1.2,.5],[bad,5],[50,5])


def test_singular_equation_rejects():
    with pytest.raises(ZeroDivisionError):checked_log_transition([.6,.3,.5,.5,1.2,.5],[-25,5],[50,5])
