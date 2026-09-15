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
    assert partial["minimum_cohort_seconds"] > config["budget"]["campaign_remaining_seconds"]
    assert partial["missing_training_scopes"]
    assert partial["full_campaign_priced"] is False
    complete = training_reservation(config, [measured_row(w, b)
        for w in config["training"]["widths"] for b in config["training"]["betas"][1:]])
    assert not complete["missing_training_scopes"]
    assert complete["minimum_cohort_seconds"] > partial["minimum_cohort_seconds"]
    assert complete["full_campaign_priced"] is False


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
