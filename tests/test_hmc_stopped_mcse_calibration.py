"""Independent reference/accounting oracles, not stochastic performance tests."""
import math
from copy import deepcopy

import pytest

from scripts.calibrate_hmc_stopped_mcse import (
    METHODS, REPAIR_METHODS, LOOKS, carry_completed_replicates, finite_mean_variance,
    interval, paired_event_difference, paired_mean, summarize, summarize_pairs)


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


def test_repair_profile_preserves_its_own_method_inventory_on_resume():
    new, old = continuation_fixture()
    for report in (new, old):
        report.update(profile="batch_maximum_repair", methods=REPAIR_METHODS, driver_sha256="same")
    for cell in old["cells"]:
        methods = cell["records"][0]["methods"]
        methods["lugsail_max_batch_means"] = methods.pop("even_lugsail")
    carry_completed_replicates(new, old)
    assert set(new["cells"][0]["summary"]["methods"]) == set(REPAIR_METHODS)
    old["driver_sha256"] = "changed"
    with pytest.raises(ValueError, match="driver"):
        carry_completed_replicates(new, old)
    old["profile"] = "historical"
    with pytest.raises(ValueError, match="profile"):
        carry_completed_replicates(new, old)


def test_paired_summary_preserves_missing_endpoints_and_variance_direction():
    original = dict(draws=10000, valid=[True, False, True], error=[0.]*3,
                    mcse=[.1, None, .3], interval_miss=[True, None, False])
    maximum = dict(draws=10000, valid=[True]*3, error=[0.]*3,
                   mcse=[.2, .3, .3], interval_miss=[False]*3)
    record = dict(methods={"lugsail": dict(fixed=original, stopped=original),
                          "lugsail_max_batch_means": dict(fixed=maximum, stopped=None)})
    summary = summarize_pairs([record])
    fixed = summary["endpoints"]["fixed"]
    assert fixed["quantities"][0]["estimated_variance_change"]["mean"] == pytest.approx(.03)
    assert fixed["quantities"][0]["interval_miss_change"]["mean"] == -1
    assert fixed["quantities"][1]["maximum_only_valid"] == 1
    assert fixed["quantities"][1]["common_valid_pairs"] == 0
    stopped = summary["endpoints"]["stopped"]
    assert stopped["common_outputs"] == 0 and stopped["original_only_output"] == 1
    assert paired_mean([])["mean"] is None
    interval = paired_mean([1., 3.])["approximate_95_percent_interval"]
    assert interval[0] < 2. < interval[1]


def test_paired_binary_uncertainty_does_not_collapse_when_every_pair_ties():
    result = paired_event_difference([0]*128)
    assert result["mean"] == 0
    lo, hi = result["conservative_95_percent_interval"]
    assert lo < 0 < hi
    assert hi == pytest.approx(-lo)
    result = paired_event_difference([-1, -1, 0, 1])
    assert result["mean"] == -.25
    assert result["maximum_only_miss"] == 1 and result["original_only_miss"] == 2
    assert paired_event_difference([])["conservative_95_percent_interval"] is None
    with pytest.raises(ValueError):
        paired_event_difference([2])


def test_heavy_tail_square_variance_difference_does_not_claim_finite_variance():
    original = dict(draws=10000, valid=[True]*3, mcse=[.1]*3, interval_miss=[False]*3)
    maximum = dict(draws=10000, valid=[True]*3, mcse=[.2]*3, interval_miss=[False]*3)
    record = dict(methods={"lugsail": dict(fixed=original, stopped=None),
                          "lugsail_max_batch_means": dict(fixed=maximum, stopped=None)})
    report = summarize_pairs([record, record], law="student_t5")
    square = report["endpoints"]["fixed"]["quantities"][1]["estimated_variance_change"]
    assert square["mean"] == pytest.approx(.03)
    assert square["approximate_95_percent_interval"] is None
    assert square["mean_role"] == "descriptive_only"
    assert "eighth moment" in square["uncertainty_status"]
    assert report["endpoints"]["fixed"]["quantities"][1]["interval_miss_change"]["conservative_95_percent_interval"][1] > 0
