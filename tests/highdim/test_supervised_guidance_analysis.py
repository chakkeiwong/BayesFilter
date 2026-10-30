"""Independent reporting checks; these tests do not run a scientific kernel."""

import pytest

from bayesfilter.score_study.supervised_guidance_analysis import summarize


def paired_rows():
    rows = []
    for seed, losses in ((10, (1., 3.)), (20, (10.,))):
        for particle, loss in enumerate(losses):
            for method, value in (("unguided", 0.), ("guided", loss)):
                rows.append(dict(partition="test", observation_seed=seed,
                    particle_seed=particle, method=method, valid=True, checks=[],
                    loss=[value, 0., 0.], terminal_moment_squared_error=[value]))
    return rows


def test_particle_replicates_are_averaged_before_datasets():
    report = summarize(paired_rows(), {})
    entry = report["paired_against_unguided"]["guided"]
    assert entry["mean_total_loss_difference"] == 6.
    assert [item["total_loss_difference"] for item in entry["datasets"]] == [2., 10.]
    assert report["inference_status"]["default_readiness"] is False


def test_invalid_replicate_prevents_passing_by_dropping_it():
    rows = paired_rows()
    rows[1].update(valid=False, checks=[{"program_valid": False}])
    report = summarize(rows, {})
    assert report["failures"]["guided"]["rows"] == 1
    assert not report["paired_against_unguided"]["guided"]["all_valid"]
    assert "mean_total_loss_difference" not in report["paired_against_unguided"]["guided"]


def test_missing_paired_row_is_rejected():
    with pytest.raises(ValueError, match="Incomplete paired"):
        summarize(paired_rows()[1:], {})
