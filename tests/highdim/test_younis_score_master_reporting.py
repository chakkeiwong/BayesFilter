"""Independent small-example checks of dataset/replicate aggregation."""
import pytest

from bayesfilter.score_study.reporting import paired_dataset_summary


def test_unequal_replication_keeps_datasets_as_sampling_units():
    # Dataset 1 has three replicates of error difference 1; dataset 2 has one
    # difference 5. Pooling replicates would incorrectly report 2 instead of 3.
    left = {(1, 0): 2., (1, 1): 3., (1, 2): 4., (2, 0): 6.}
    right = {(1, 0): 1., (1, 1): 2., (1, 2): 3., (2, 0): 1.}
    result = paired_dataset_summary(left, right)
    assert result["mean_squared_error_difference"] == pytest.approx(3.)
    assert result["dataset_mcse"] == pytest.approx(2.)
    assert not result["statistically_supported_ranking"]


def test_missing_replicate_is_a_coverage_veto_not_complete_case_selection():
    result = paired_dataset_summary({(1, 0): 1., (2, 0): 9.}, {(1, 0): 1.})
    assert result["status"] == "coverage_veto"


def test_common_error_cancels_and_one_dataset_does_not_invent_mcse():
    values = {(1, 0): 2., (1, 1): 100.}
    result = paired_dataset_summary(values, values)
    assert result["mean_squared_error_difference"] == 0.
    assert result["dataset_mcse"] is None
