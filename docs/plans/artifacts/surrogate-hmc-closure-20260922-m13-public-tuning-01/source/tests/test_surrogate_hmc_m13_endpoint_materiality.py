"""Focused diagnostic contracts for same-data endpoint uncertainty."""
import math
import pytest
import tensorflow as tf
from bayesfilter.testing.surrogate_hmc_m13_endpoint_materiality import (
    PLAN, accepted_path, likelihood_statistics, pair_statistics)
from bayesfilter.inference.surrogate_hmc_closure_program import jobs, PASS_CAPS


def test_shared_multiplicative_error_cancels_from_log_likelihood_ratio():
    logs = tf.math.log(tf.constant([[1.,2.,3.,4.], [7.,14.,21.,28.]], tf.float64))
    means, mcse, _ = likelihood_statistics(logs)
    difference, error = pair_statistics(logs, tf.constant([[0,1]], tf.int32))
    tf.debugging.assert_near(difference, [tf.constant(math.log(7.), tf.float64)], atol=1e-12)
    assert float(error[0]) < 1e-12
    assert float(mcse[0]) > 0.0
    # Explicit sample variance: normalized influences [-.6,-.2,.2,.6].
    expected = math.sqrt((.36+.04+.04+.36)/12)
    assert float(mcse[0]) == pytest.approx(expected)


def test_constant_log_offset_does_not_change_mcse_or_pair_difference():
    logs=tf.constant([[1.,2.,3.], [2.,1.,4.]], tf.float64)
    pairs=tf.constant([[0,1]],tf.int32)
    a=pair_statistics(logs,pairs); b=pair_statistics(logs+1234.,pairs)
    tf.debugging.assert_near(a[0],b[0],atol=1e-10)
    tf.debugging.assert_near(a[1],b[1],atol=1e-10)


def test_rejected_transition_cannot_be_reconstructed_as_proposal():
    row=dict(status="M13_SURROGATE_HMC_MECHANICS_PASSED",hmc=dict(
        loop_jit_compile=True, positions=[[[0.,0.]]]*24,accepted=[[True]]*24))
    assert len(accepted_path(row))==25
    row['hmc']['accepted'][5]=[False]
    with pytest.raises(ValueError,match='accepted'):
        accepted_path(row)


def test_master_stage_is_bounded_and_separate_from_posterior_sampling():
    selected=jobs('m13-endpoint-materiality')
    assert [x['name'] for x in selected]==['review','m13-endpoint-materiality']
    assert selected[1]['plan']==PLAN
    assert selected[1]['device']=='gpu'
    assert PASS_CAPS['m13-endpoint-materiality']==dict(cpu_worker_seconds=1600.,gpu_seconds=1600.)
