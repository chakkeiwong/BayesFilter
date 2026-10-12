"""Independent reference/accounting oracles, not stochastic performance tests."""
import math
from copy import deepcopy

import pytest

from scripts.calibrate_hmc_stopped_mcse import (
    METHODS, LOOKS, carry_completed_replicates, finite_mean_variance, interval, summarize)


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


def continuation_fixture():
    new = dict(seed=7, replications=2, methods=METHODS, looks=LOOKS, chains=4,
        source_sha256={"source": "hash"}, source={"git_commit": "frozen"}, versions={"tf": "test"},
        stationary_initialization=True, gpu_intentionally_hidden=True, use_xla=True,
        cells=[dict(law=law, rho=rho, planned=2, records=[])
               for rho in (0., .9, -.9) for law in ("gaussian", "student_t5")])
    old = deepcopy(new)
    old.update(status="budget_exhausted", elapsed_seconds=11.)
    endpoint = dict(draws=10000, valid=[True]*3, error=[.1]*3, mcse=[.2]*3,
                    interval_miss=[False]*3)
    for index, cell in enumerate(old["cells"]):
        cell["records"] = [dict(replication=0, seed=[7, index//2, 0],
            methods={method: dict(fixed=deepcopy(endpoint), stopped=None) for method in METHODS})]
    return new, old


def test_resume_keeps_all_original_records_and_their_compute_charge():
    new, old = continuation_fixture()
    original = deepcopy(old)
    carry_completed_replicates(new, old)
    assert old == original
    assert new["carried_completed_records"] == [1]*6
    assert new["prior_elapsed_seconds"] == 11.
    assert new["attempt_count"] == 2
    assert [c["records"] for c in new["cells"]] == [c["records"] for c in old["cells"]]


@pytest.mark.parametrize("mutation", ["source", "seed", "count", "law", "duplicate", "nonprefix",
                                      "incomplete", "wrong_stream", "running", "version"])
def test_resume_rejects_changed_design_or_incomplete_completed_records(mutation):
    new, old = continuation_fixture()
    if mutation == "source": new["source_sha256"]["source"] = "new"
    elif mutation == "seed": new["seed"] = 8
    elif mutation == "count": new["replications"] = 3
    elif mutation == "law": new["cells"][0]["law"] = "new-law"
    elif mutation == "duplicate": old["cells"][0]["records"] *= 2
    elif mutation == "nonprefix": old["cells"][0]["records"][0]["replication"] = 1
    elif mutation == "incomplete": old["cells"][0]["records"][0]["methods"]["lugsail"]["fixed"] = None
    elif mutation == "wrong_stream": old["cells"][0]["records"][0]["seed"] = [7, 1, 0]
    elif mutation == "running": old["status"] = "running"
    elif mutation == "version": new["versions"]["tf"] = "changed"
    with pytest.raises(ValueError):
        carry_completed_replicates(new, old)
