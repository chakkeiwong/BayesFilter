"""Independent paper identities and actual public-R call-chain diagnostics.

These tests passing means the comparator is classified correctly. They do not
mean its known departures from the paper have been repaired or admitted.
"""
from __future__ import annotations

import copy
import math
import shutil

import pytest

from docs.benchmarks import diagnose_iapf_paper_conformance as audit
from docs.benchmarks import diagnose_younis_iapf_kdm_reference_comparison as comparison


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    if (not shutil.which("Rscript") or not audit.SOURCE.exists()
            or not audit.PAPER.exists()):
        pytest.skip("external pinned iAPF source, paper and R required; no conformance admission issued")
    return audit.run_reference(tmp_path_factory.mktemp("iapf-paper-source"))


@pytest.fixture(scope="module")
def report(reference):
    return audit.assess_reference(reference)


def check(report, name):
    return next(row for row in report["checks"] if row["name"] == name)


@pytest.mark.external
@pytest.mark.parametrize("name", sorted(audit.CORE))
def test_executed_reference_satisfies_independent_gaussian_identity(report, name):
    assert check(report, name)["passed"], check(report, name)


def test_full_paper_admission_rejected_despite_matching_shared_operations(report):
    audit.require_reference(report, scope="matched_gaussian")
    assert set(report["deviations"]) == {
        "outer_stopping_index", "paper_fit_objective", "positive_floor", "single_observation_domain"}
    assert report["paper_reference_eligible"] is False
    assert report["original_author_provenance"] == "not_verified"
    with pytest.raises(ValueError, match="paper_fit_objective"):
        audit.require_reference(report)


def test_actual_objective_is_transformed_shape_loss_not_equation_15(report):
    objective = check(report, "paper_fit_objective")
    assert not objective["passed"]
    for actual, relation in zip(objective["actual"], report["objective_relations"]):
        assert math.isclose(actual, relation["public_objective_from_shape"], abs_tol=2e-10, rel_tol=2e-10)
    assert abs(objective["actual"][0]-objective["expected"][0]) > 1e-3


def test_actual_twist_omits_declared_positive_floor(report):
    row = check(report, "positive_floor")
    assert not row["passed"]
    assert all(math.isclose(b-a, .01, abs_tol=1e-14) for a,b in zip(row["actual"],row["expected"]))
    assert row["layer"] == "example_scheme"  # Not a general twisted-PF invalidity claim.


def test_source_outer_loop_stops_one_estimate_early_on_stable_history(report):
    row = check(report, "outer_stopping_index")
    assert row["actual"][0] == 2  # Source l=3 maps to paper l=2.
    assert row["expected"][0] == 3  # Paper requires l>k, with k=2.
    assert check(report, "fresh_final_run")["passed"]


def test_single_observation_is_a_source_domain_failure(report):
    row = check(report, "single_observation_domain")
    assert row["actual"] == [0.]
    assert row["passed"] is False


@pytest.mark.parametrize("damage", ["missing", "empty", "duplicate", "schema", "source", "partial"])
def test_admission_rejects_incomplete_or_drifted_evidence(report, damage):
    damaged = copy.deepcopy(report)
    if damage == "missing":
        damaged["checks"] = [row for row in damaged["checks"] if row["name"] != "path_density_identity"]
    elif damage == "empty":
        damaged["checks"] = []
    elif damage == "duplicate":
        damaged["checks"].append(damaged["checks"][0])
    elif damage == "schema":
        damaged["schema"] = "legacy_component_comparison"
    elif damage == "source":
        damaged["source_sha256"] = "unreviewed"
    else:
        damaged["execution_status"] = "incomplete"
    with pytest.raises(ValueError, match="incomplete or unrecognized"):
        audit.require_reference(damaged, scope="matched_gaussian")


def test_a_claim_field_cannot_override_failed_paper_requirements(report):
    damaged = copy.deepcopy(report)
    damaged["paper_reference_eligible"] = True
    with pytest.raises(ValueError, match="does not meet paper"):
        audit.require_reference(damaged)


@pytest.mark.external
@pytest.mark.parametrize("old,new,failed_check", [
    ("return(g(y, x)*psi_tilda(x, psi_pa, 1)*", "return(2*g(y, x)*psi_tilda(x, psi_pa, 1)*", "path_density_identity"),
    ("w[t,i] <- w[t-1,i] + log(g_aux(", "w[t,i] <- log(g_aux(", "apf_retain"),
    # The ideal twist has unit terminal weights, so only the nonideal fixture
    # can detect removal of its terminal likelihood contribution.
    ("Z[l] <- Z[l] + log(mean(exp(w[t,]-mx))) + mx", "Z[l] <- Z[l] + 0", "apf_retain"),
])
def test_mathematical_checks_detect_mutated_executed_source(reference, tmp_path, old, new, failed_check):
    source_text = audit.SOURCE.read_text()
    assert source_text.count(old) == 1
    changed_source = tmp_path/"changed-iapf.R"
    changed_source.write_text(source_text.replace(old,new))
    changed = audit.assess_reference(audit.run_reference(tmp_path/"probe", changed_source), changed_source)
    assert not check(changed, "source_identity")["passed"]
    # A provenance mismatch alone would not demonstrate mathematical detection.
    assert not check(changed, failed_check)["passed"], check(changed, failed_check)


def test_comparison_consumer_uses_conformance_before_tensorflow(reference, tmp_path, monkeypatch):
    damaged = copy.deepcopy(reference)
    damaged["paper_retain"]["log_likelihood"][0] += 1.
    monkeypatch.setattr(audit, "run_reference", lambda *args, **kwargs: damaged)
    # Actual consumer must reject before touching the supplied non-TF sentinel.
    with pytest.raises(ValueError, match="apf_retain"):
        comparison.run_iapf(tf=None, checks=None, output=tmp_path)
    assert (tmp_path/"iapf-paper-conformance.json").exists()


def test_comparison_preparation_preserves_restricted_reference_scope(reference, tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "run_reference", lambda *args, **kwargs: reference)
    _, evidence = comparison.prepare_iapf_reference(tmp_path)
    assert evidence["paper_reference_eligible"] is False
    assert evidence["reference_scope"] == "matched_gaussian_operations_only"


def test_csv_reader_rejects_duplicate_observations(tmp_path):
    path = tmp_path/"corrupt.csv"
    path.write_text("case,field,index,value\nx,y,1,2\nx,y,1,3\n")
    with pytest.raises(ValueError, match="duplicate"):
        audit.read_reference(path)
