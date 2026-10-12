"""Independent reference/accounting oracles, not stochastic performance tests."""
import math

import pytest

from scripts.calibrate_hmc_stopped_mcse import METHODS, finite_mean_variance, interval, summarize


@pytest.mark.parametrize("rho",[0.,.9,-.9])
@pytest.mark.parametrize("n",[1,5,40])
def test_exact_variance_matches_full_covariance_sum(rho,n):
    direct=math.fsum(rho**abs(i-j) for i in range(n) for j in range(n))/(4*n*n)
    assert finite_mean_variance(n,rho)==pytest.approx(direct,rel=1e-12)


def test_invalid_intervals_and_nonstoppers_remain_in_denominator():
    endpoint=dict(draws=10000,valid=[True,False,True],error=[.1,.2,.3],mcse=[.2,None,.1],
                  interval_miss=[False,None,True])
    row=dict(methods={m:dict(fixed=endpoint,stopped=None) for m in METHODS})
    result=summarize([row],128)
    assert result["unstarted"]==127
    for entry in result["methods"].values():
        assert entry["stopped"]==0
        square=entry["endpoints"]["fixed"]["quantities"][1]
        assert square["missing_intervals"]==1 and square["valid_intervals"]==0
        assert square["miss_rate_among_valid_interval"] is None
        assert square["empirical_mean_squared_error"]==pytest.approx(.04)
        assert entry["endpoints"]["stopped"]["quantities"][0]["output_available"]==0
    assert interval(0,0) is None
    assert interval(0,128)[1]>0
