"""Independent arithmetic/record corruption checks for the saved diagnostic."""
import math
import pytest
from scripts.diagnostics.audit_modern_hmc_saved_precision import (
    check_endpoint, replication_summary, stationary_variance,
    insufficient_quantities,
)

@pytest.mark.parametrize("rho", [0., .99, -.99])
@pytest.mark.parametrize("power", [1, 2])
def test_stationary_variance_matches_full_covariance_matrix(rho, power):
    n, m = 17, 4
    factor = 1. if power == 1 else 2.
    exact = math.fsum(factor*rho**(power*abs(i-j))
                      for i in range(n) for j in range(n))/(m*n*n)
    assert stationary_variance(n,m,rho,power) == pytest.approx(exact,rel=1e-12)

def test_mean_variance_respects_chain_count_and_translation():
    result = replication_summary([1.,2.,3.,4.])
    assert result["variance"] == pytest.approx(5./12)
    shifted = replication_summary([101.,102.,103.,104.])
    assert shifted["variance"] == result["variance"]
    assert shifted["estimate"] == result["estimate"]+100

def endpoint():
    means = [[-1.,0.],[1.,2.]]
    quantities=[];targets=[]
    for i,name in enumerate(["coordinate","square"]):
        r=replication_summary([x[i] for x in means])
        quantities.append(dict(name=name,estimate=r["estimate"],mcse=r["mcse"],
            absolute_tolerance=.5,precision_passed=False,truth=r["estimate"],
            multiplier=2.,covered=True))
        targets.append(dict(name=name,estimate=r["estimate"],mcse=r["mcse"],
            mcse_absolute_max=.5,passed=False))
    return dict(chain_means=means,quantities=quantities,report=dict(targets=targets))

def test_failed_precision_stays_failed_after_reduction_audit():
    assert all(not x["precision_passed"] for x in check_endpoint(endpoint(),2))

@pytest.mark.parametrize("tamper",["mcse","decision","mean","threshold"])
def test_corrupt_stored_evidence_is_rejected(tamper):
    e=endpoint()
    if tamper=="mcse": e["quantities"][0]["mcse"]*=2
    elif tamper=="decision":e["quantities"][0]["precision_passed"]=True
    elif tamper=="mean":e["chain_means"][0][0]+=1
    else:e["report"]["targets"][0]["mcse_absolute_max"]=2
    with pytest.raises(ValueError):check_endpoint(e,2)

def test_warmup_refusal_is_not_a_missing_precision_failure():
    assert insufficient_quantities(dict(failed_checks=["warmup"],passed=False)) == []
    with pytest.raises(ValueError,match="unexplained missing precision"):
        insufficient_quantities(dict(failed_checks=[],passed=True))
