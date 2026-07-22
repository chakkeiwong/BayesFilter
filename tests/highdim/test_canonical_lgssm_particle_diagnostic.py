from __future__ import annotations

from docs.benchmarks import aggregate_canonical_lgssm_particle_diagnostic as aggregate
from docs.benchmarks import run_canonical_lgssm_particle_diagnostic_arm as arm
import pytest

from bayesfilter.highdim.transport_chunk_policy import TRANSPORT_CHUNK_POLICY_ID


def test_particle_ladder_and_seeds_are_frozen() -> None:
    assert arm.ALLOWED_PARTICLE_COUNTS == (128, 256, 512, 1024)
    assert arm.ESTIMATOR_SEEDS == tuple(range(81400, 81416))
    assert arm.BALANCE_STEPS == 50
    assert arm.TIME_STEPS == 2


def test_screen_classification_is_exact() -> None:
    inside = [
        {"lower": -margin / 2.0, "upper": margin / 2.0}
        for margin in aggregate.MARGINS
    ]
    outside = list(inside)
    outside[3] = {"lower": 0.06, "upper": 0.08}
    overlap = list(inside)
    overlap[2] = {"lower": -0.04, "upper": 0.06}
    assert aggregate._screen(inside, True) == "screen_pass"
    assert aggregate._screen(outside, True) == "screen_fail"
    assert aggregate._screen(overlap, True) == "inconclusive"
    assert aggregate._screen(inside, False) == "screen_fail"


def test_interval_uses_frozen_bonferroni_student_critical_value() -> None:
    values = [float(index) for index in range(16)]
    interval = aggregate._interval(values)
    assert interval["mean"] == 7.5
    assert interval["critical_value"] == 3.036283222821165
    assert interval["lower"] < interval["mean"] < interval["upper"]


def test_aggregate_rejects_prepolicy_or_wrong_chunk_identity() -> None:
    valid = {
        "transport_chunk_policy_id": TRANSPORT_CHUNK_POLICY_ID,
        "row_chunk_size": 128,
        "col_chunk_size": 128,
        "transport_block_grid": [1, 1],
    }
    aggregate._require_transport_chunk_identity(valid, 128)
    with pytest.raises(ValueError, match="ineligible transport chunk identity"):
        aggregate._require_transport_chunk_identity({}, 128)
    with pytest.raises(ValueError, match="ineligible transport chunk identity"):
        aggregate._require_transport_chunk_identity(
            {**valid, "row_chunk_size": 16, "col_chunk_size": 16}, 128
        )
