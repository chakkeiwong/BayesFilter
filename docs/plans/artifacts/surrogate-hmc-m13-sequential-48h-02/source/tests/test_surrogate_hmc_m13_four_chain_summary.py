"""Diagnostic saved-chunk recovery validates failures as well as good chunks."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pytest
import tensorflow as tf
from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import (
    STARTS, batched_endpoint, bind_components, compiled_chunk,
)
from bayesfilter.testing.surrogate_hmc_m13_four_chain_summary import assess_chunk, mechanics
from bayesfilter.inference.surrogate_hmc_closure_program import jobs


def test_saved_chunk_health_rechecks_instead_of_trusting_saved_flag():
    def evaluator(x):
        return -tf.reduce_sum(x*x), tf.constant(True), {}
    endpoint = batched_endpoint(evaluator)
    binding = bind_components(lambda x: x, endpoint, "summary-test")
    run, _, _ = compiled_chunk(binding, "summary-test", 2)
    chunk = run(tf.constant(STARTS, tf.float64), tf.constant([321, 123], tf.int32))
    record = mechanics.serial(chunk._asdict())
    assert assess_chunk(record, STARTS)["passed"]
    record["force_fallback"][0][0] = True
    assert not assess_chunk(record, STARTS)["passed"]
    record["force_fallback"][0][0] = False
    record["endpoint_status"][0][0][8] = 1
    assert not assess_chunk(record, STARTS)["passed"]
    record["states"] = record["states"][:-1]
    with pytest.raises(ValueError, match="shape"):
        assess_chunk(record, STARTS)


def test_summary_only_cpu_master_stage():
    selected = jobs("m13-four-chain-summary")
    assert [x["name"] for x in selected] == ["review", "m13-four-chain-summary"]
    assert all(x["device"] == "cpu" for x in selected)


def test_zero_gpu_budget_allowed_only_for_explicit_cpu_mode():
    from bayesfilter.inference.surrogate_hmc_program import validated_budget
    from bayesfilter.inference.surrogate_hmc_closure_program import remaining_pass_reservation
    with pytest.raises(ValueError):
        validated_budget(0., 120.)
    assert validated_budget(0., 120., allow_zero_gpu=True)["limits"]["gpu_seconds"] == 0.
    assert remaining_pass_reservation([], "m13-four-chain-summary")["gpu_seconds"] == 0.
    with pytest.raises(ValueError, match="exhausted"):
        remaining_pass_reservation([dict(path="surrogate-hmc-closure-20260922-m13-four-chain-summary-01/accounting.json",
            charged_this_run=dict(cpu_worker_seconds=120., gpu_seconds=0.))], "m13-four-chain-summary")
