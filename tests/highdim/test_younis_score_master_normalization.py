"""Independent exact witnesses for normalization and evidence-joining diagnostics."""
import copy
import math

import pytest

from bayesfilter.score_study import normalization_reporting as report


def test_covariance_and_ratio_of_means_are_distinct_from_average_score():
    result = report.normalization_summary([0., math.log(3.)], [[2.], [4.]], 0., [3.])
    assert result["mean_scaled_normalizer"] == pytest.approx(2.)
    assert result["mean_scaled_derivative"] == pytest.approx([7.])
    assert result["mean_individual_log_score"] == pytest.approx([3.])
    assert result["ratio_of_means"] == pytest.approx([3.5])
    assert result["normalizer_score_covariance_divisor_R"] == pytest.approx([1.])
    assert result["covariance_identity_residual"] == pytest.approx([0.], abs=1e-14)
    assert result["ratio_of_means_bias_expansion_plugin"] == pytest.approx([-.375])
    assert result["empirical_score_mse"] == pytest.approx(1.)
    assert result["score_sample_variance_trace"] == pytest.approx(2.)


def test_unbiased_likelihood_derivative_does_not_make_individual_log_score_unbiased():
    # Zhat=theta+X at theta=2, equiprobable X=-1,+1; Dhat=1 exactly.
    result = report.normalization_summary([0., math.log(3.)], [[1.], [1/3]], math.log(2.), [.5])
    assert result["mean_scaled_normalizer"] == pytest.approx(1.)
    assert result["mean_scaled_derivative"] == pytest.approx([.5])
    assert result["ratio_of_means"] == pytest.approx([.5])
    assert result["mean_individual_log_score"] == pytest.approx([2/3])
    assert result["empirical_score_bias"] == pytest.approx([1/6])
    assert not result["unbiasedness_established"]


def test_constant_normalizer_and_zero_variance_do_not_invent_signal():
    result = report.normalization_summary([.123456789]*3, [[1.], [2.], [3.]], .123456789, [2.])
    assert result["mean_scaled_normalizer"] == 1.
    assert result["ratio_of_means_bias_expansion_plugin"] == [0.]
    assert result["normalizer_cv"] == 0.
    assert report.descriptive_correlation([0., 0.], [1., 2.])["correlation"] is None
    assert report.descriptive_correlation([1., 2., 3.], [3., 2., 1.])["correlation"] == pytest.approx(-1.)


@pytest.mark.parametrize("logs,scores", [([0.], [[1.]]), ([0., 1.], [[1.]]), ([0., 800.], [[1.], [1.]])])
def test_invalid_normalization_inputs_fail_closed(logs, scores):
    with pytest.raises(ValueError):
        report.normalization_summary(logs, scores, 0., [0.])


def _fixture(n):
    state = {"study": {"settings": {"particles": n, "horizon": 3}, "seed": 1,
                       "partitions": {}, "evidence_class": "mechanics"}}
    records = []
    for rep in range(3):
        key = ("model", "condition", "prior_sis", "analytical", "model_score", "mechanics", "config")
        row = {"dataset": 1, "replicate": rep}
        result = {"score": [(rep+1)/n], "oracle_score": [0.], "oracle_value": 0.,
                  "diagnostics": {"data_version": "common"}}
        records.append((key, row, result))
    return state, records


def test_actual_assembly_joins_by_particle_count_and_preserves_dataset_group(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "_validated_rows", lambda path: _fixture(int(path)))
    monkeypatch.setattr("bayesfilter.score_study.runtime.configure_runtime", lambda **kwargs: {})
    result = report.assemble_consistency(["32", "8", "16"], tmp_path/"consistency.json")
    assert result["particle_counts"] == [8, 16, 32]
    assert result["groups"][0]["per_dataset"][0]["correlation"] == pytest.approx(1.)
    assert len(result["groups"][0]["rows"]) == 3
    assert not result["statistically_supported_ranking"]


@pytest.mark.parametrize("mismatch", ["scope", "data", "coverage", "configuration"])
def test_assembly_rejects_invalid_pairing(monkeypatch, tmp_path, mismatch):
    def reader(path):
        state, records = copy.deepcopy(_fixture(int(path)))
        if int(path) == 16:
            if mismatch == "scope":
                state["study"]["settings"]["horizon"] = 9
            if mismatch == "data":
                records[0][2]["diagnostics"]["data_version"] = "different"
            if mismatch == "coverage":
                records.pop()
            if mismatch == "configuration":
                key, row, result = records[0]
                records[0] = ((*key[:-1], "changed"), row, result)
        return state, records
    monkeypatch.setattr(report, "_validated_rows", reader)
    monkeypatch.setattr("bayesfilter.score_study.runtime.configure_runtime", lambda **kwargs: {})
    with pytest.raises(ValueError):
        report.assemble_consistency(["8", "16", "32"], tmp_path/"rejected.json")
