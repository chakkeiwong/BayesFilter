"""Diagnostic tests of independent-dataset uncertainty and untouched joins."""
import copy
import pytest

from bayesfilter.score_study import heldout_reporting as reporting


def test_bootstrap_keeps_dataset_as_unit_with_unequal_replication():
    left = {(1, 0): 2., (1, 1): 2., (1, 2): 2., (2, 0): 6.}
    right = {k: 1. for k in left}
    result = reporting.paired_bootstrap_summary(left, right, resamples=1000)
    assert result["mean_squared_error_difference"] == pytest.approx(3.)
    assert result["dataset_mcse"] == pytest.approx(2.)
    assert result["bootstrap_interval"] == [1., 5.]
    assert not result["statistically_supported_ranking"]


def test_pairing_cancels_shared_noise_without_inventing_uncertainty():
    values = {(i, j): float((i+1)**2+j) for i in range(4) for j in range(3)}
    result = reporting.paired_bootstrap_summary(values, values, resamples=100)
    assert result["bootstrap_interval"] == [0., 0.]
    missing = dict(values)
    missing.pop((0, 0))
    assert reporting.paired_bootstrap_summary(values, missing)["status"] == "coverage_veto"


def fixture_records(method):
    state = {"study": {"evidence_class": "research_pilot", "seed": 123,
                       "settings": {"dimension": 1, "particles": 8}}}
    records = []
    for dataset in (400, 401):
        row = {"id": method+str(dataset), "role": "claim", "condition": "curved",
               "model": "nonlinear_scalar", "comparison_target": "model_score",
               "proposal": method, "dataset": dataset, "replicate": 0}
        result = {"score": [2. if method == "ledh" else 1.]*6,
                  "oracle_score": [0.]*6, "oracle_value": -1.,
                  "runtime": {"kernel_wall_seconds": 1.},
                  "diagnostics": {"data_version": str(dataset),
                                  "candidate_configuration": {"steps": 2} if method == "ledh" else None}}
        records.append(((), row, result))
    return state, records


def contract():
    return {"datasets": [400, 401], "replicates": [0], "conditions": ["curved"],
            "methods": ["ledh", "ekf"], "candidates": ["ledh"], "heuristics": ["ekf"],
            "bootstrap_resamples": 100, "bootstrap_seed": 123}


def test_join_reports_heuristic_loss_and_all_six_coordinates(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "_validated_rows", fixture_records)
    result = reporting.assemble_heldout(["ledh", "ekf"], tmp_path/"report.json", contract())
    assert result["heuristic_dominance_verdict"] == "promotion_veto_observed_losses"
    assert result["paired_comparisons"][0]["mean_squared_error_difference"] == pytest.approx(18.)
    assert result["summaries"][1]["coordinate_mse"] == [4.]*6
    assert not result["default_ready"]


@pytest.mark.parametrize("corruption", ["data", "oracle", "scope", "missing", "role", "config", "duplicate"])
def test_join_rejects_scientific_mismatches(tmp_path, monkeypatch, corruption):
    def corrupted(method):
        state, rows = copy.deepcopy(fixture_records(method))
        if method == "ledh":
            if corruption == "data": rows[0][2]["diagnostics"]["data_version"] = "wrong"
            if corruption == "oracle": rows[0][2]["oracle_score"][0] = 1.
            if corruption == "scope": state["study"]["settings"]["particles"] = 16
            if corruption == "missing": rows.pop()
            if corruption == "role": rows[0][1]["role"] = "calibration"
            if corruption == "config": rows[0][2]["diagnostics"]["candidate_configuration"] = {"steps": 8}
            if corruption == "duplicate": rows.append(copy.deepcopy(rows[0]))
        return state, rows
    monkeypatch.setattr(reporting, "_validated_rows", corrupted)
    with pytest.raises(ValueError):
        reporting.assemble_heldout(["ledh", "ekf"], tmp_path/"report.json", contract())
