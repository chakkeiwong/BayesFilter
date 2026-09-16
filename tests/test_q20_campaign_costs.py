"""Cost-admission checks; no GPU or inference claim."""
import copy

import pytest

from bayesfilter.inference.q20_campaign_costs import training_reservation
from bayesfilter.inference.q20_production_config import protocol_template


def measured_row(width=16, beta=.5):
    return {"width": width, "beta": beta, "batch_size": 32,
            "first_update_seconds": 12., "steady_update_seconds": 3.,
            "heldout_first_seconds": 13., "heldout_seconds_per_batch": 3.}


def test_partial_cost_can_reject_but_cannot_admit_campaign():
    config = protocol_template()
    partial = training_reservation(config, [measured_row()])
    assert partial["minimum_cohort_seconds"] > 0
    assert partial["missing_training_scopes"]
    assert partial["full_campaign_priced"] is False
    complete = training_reservation(config, [measured_row(w, b)
        for w in config["training"]["widths"] for b in config["training"]["betas"][1:]])
    assert not complete["missing_training_scopes"]
    assert complete["minimum_cohort_seconds"] > partial["minimum_cohort_seconds"]
    assert complete["full_campaign_priced"] is False
    scenarios = complete["scenarios"]
    assert scenarios["calibration"]["training_scopes"] == 8
    assert scenarios["calibration"]["optimizer_updates"] == 8 * 128
    assert scenarios["calibration"]["target_validation_rows"] == 8 * 2 * 768
    assert scenarios["floor_validation_cap"]["optimizer_updates"] == 36 * 512
    assert scenarios["floor_first_bank"]["target_validation_rows"] == 36 * 3 * 768
    assert scenarios["floor_validation_cap"]["target_validation_rows"] == 36 * 3 * 12288
    assert scenarios["full_cap"]["target_validation_rows"] == 36 * 5 * 12288
    # In particular, no duplicate baseline/previous graph at the first rung.
    assert scenarios["calibration"]["optimizer_seconds"] == 8 * (12. + 127 * 3.)
    assert scenarios["calibration"]["validation_seconds"] == 16 * (13. - 2 * 3. + 24 * 3.)
    assert complete["calibration_seconds"] < complete["minimum_cohort_seconds"] < complete["full_training_cap_seconds"]


@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf")])
def test_invalid_measurements_cannot_issue_affordability_verdict(bad):
    row = measured_row()
    row["steady_update_seconds"] = bad
    with pytest.raises(ValueError, match="finite and positive"):
        training_reservation(protocol_template(), [row])


def test_duplicate_or_wrong_scope_cannot_inflate_reservation():
    row = measured_row()
    with pytest.raises(ValueError, match="duplicate or unsupported"):
        training_reservation(protocol_template(), [row, copy.deepcopy(row)])
    row["width"] = 999
    with pytest.raises(ValueError, match="duplicate or unsupported"):
        training_reservation(protocol_template(), [row])


def test_invalid_setup_or_batch_cost_metadata_rejects():
    for field, value in (("setup_seconds", float("nan")), ("heldout_first_batches", 0)):
        with pytest.raises(ValueError):
            training_reservation(protocol_template(), [{**measured_row(), field: value}])
