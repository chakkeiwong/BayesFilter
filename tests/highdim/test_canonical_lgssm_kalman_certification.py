from __future__ import annotations

import math
import inspect

import tensorflow as tf
import pytest

from docs.benchmarks import aggregate_canonical_lgssm_kalman_certification as aggregate
from docs.benchmarks import run_canonical_lgssm_kalman_certification_arm as arm
from docs.benchmarks import run_canonical_lgssm_same_scalar_fd as fd
from docs.benchmarks import emit_contract_e_canonical_lgssm_phase8_target_prefix_smoke
from docs.benchmarks import run_contract_e_phase8_lower_rung_node
from docs.benchmarks import run_contract_e_phase8_paired_audit16
from bayesfilter.highdim.transport_chunk_policy import TRANSPORT_CHUNK_POLICY_ID


def test_certification_design_is_frozen() -> None:
    assert arm.NUM_PARTICLES == 1024
    assert arm.BALANCE_STEPS == 50
    assert arm.ESTIMATOR_SEEDS_BY_HORIZON == {
        2: tuple(range(81500, 81516)),
        10: tuple(range(81520, 81536)),
        50: tuple(range(81540, 81556)),
    }
    assert arm.GPU_MEMORY_LIMIT_MIB == 8192


def test_certification_screen_uses_only_kalman_margins() -> None:
    inside = [
        {"lower": -margin / 2.0, "upper": margin / 2.0}
        for margin in aggregate.MARGINS
    ]
    overlap = list(inside)
    overlap[0] = {"lower": -0.0005, "upper": 0.0015}
    outside = list(inside)
    outside[4] = {"lower": 0.06, "upper": 0.08}
    assert aggregate._screen(inside, True) == "screen_pass"
    assert aggregate._screen(overlap, True) == "inconclusive"
    assert aggregate._screen(outside, True) == "screen_fail"
    assert aggregate._screen(inside, False) == "screen_fail"


def test_fd_node_uses_reviewed_dyadic_hmc_coordinate_step() -> None:
    assert fd.FD_STEP == 2.0**-17
    assert fd.FD_STEP.hex() == "0x1.0000000000000p-17"
    physical = tf.constant(fd.THETA, tf.float64)
    hmc = tf.concat([tf.math.atanh(physical[:3]), tf.math.log(physical[3:])], axis=0)
    recovered = fd._physical_from_hmc(hmc)
    tf.debugging.assert_near(recovered, physical, atol=2.0e-16, rtol=0.0)
    assert math.isfinite(float(tf.reduce_sum(hmc).numpy()))


def test_historical_preparation_callers_do_not_misstate_executed_balance_count() -> None:
    for module in (
        emit_contract_e_canonical_lgssm_phase8_target_prefix_smoke,
        run_contract_e_phase8_lower_rung_node,
        run_contract_e_phase8_paired_audit16,
    ):
        source = inspect.getsource(module)
        assert "balance_steps=0" not in source
        assert "balance_steps=1" in source


def test_certification_rejects_prepolicy_or_wrong_chunk_identity() -> None:
    valid = {
        "transport_chunk_policy_id": TRANSPORT_CHUNK_POLICY_ID,
        "row_chunk_size": 1024,
        "col_chunk_size": 1024,
        "transport_block_grid": [1, 1],
    }
    aggregate._require_transport_chunk_identity(valid, 1024)
    with pytest.raises(ValueError, match="ineligible transport chunk identity"):
        aggregate._require_transport_chunk_identity({}, 1024)
    with pytest.raises(ValueError, match="ineligible transport chunk identity"):
        aggregate._require_transport_chunk_identity(
            {**valid, "row_chunk_size": 16, "col_chunk_size": 16}, 1024
        )
