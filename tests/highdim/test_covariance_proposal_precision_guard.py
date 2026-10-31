"""CPU diagnostic: exercise the precision guard through the real filter consumer."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pytest
import tensorflow as tf

import bayesfilter.highdim.covariance_proposal_tf as runtime
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec


def evaluate(monkeypatch, dtype, covariance_error=0., failure=None):
    original = runtime.reset_cloud

    def perturbed(*args):
        z, dz, mu, dmu, p, dp, valid, diagnostics = original(*args)
        scale = tf.sqrt(tf.constant(1. + covariance_error, dtype))
        z, dz = mu + scale*(z-mu), dmu + scale*(dz-dmu)
        if failure == 'nonfinite':
            dz = tf.fill(tf.shape(dz), tf.constant(float('nan'), dtype))
        if failure == 'covariance_nonfinite':
            p = tf.fill(tf.shape(p), tf.constant(float('nan'), dtype))
        if failure == 'rank':
            valid = tf.constant(False)
        if failure == 'mean':
            z = z + tf.constant(.02, dtype)
        return z, dz, mu, dmu, p, dp, valid, diagnostics

    monkeypatch.setattr(runtime, 'reset_cloud', perturbed)
    spec = LGSSMSpec('diagonal_ar', 1)
    program = runtime.make_filter(spec, 32, 1, runtime.ProposalControls(), dtype,
                                  jit_compile=False)
    theta = tf.constant([.9, .6, .8], dtype)
    return program(theta, tf.constant([1., 0., 0.], dtype),
                   tf.constant([[.5]], dtype),
                   tf.random.stateless_normal([32, 1], [41, 1], dtype=dtype),
                   tf.random.stateless_normal([1, 32, 1], [41, 2], dtype=dtype),
                   tf.random.stateless_uniform([1, 32, 2], [41, 3], dtype=dtype),
                   tf.constant([.2, .3, .5], dtype))


@pytest.mark.parametrize('error,accepted', [(.009, True), (.011, False)])
def test_fp32_covariance_one_percent_boundary(monkeypatch, error, accepted):
    result = evaluate(monkeypatch, tf.float32, error)
    assert bool(result['valid']) is accepted
    assert int(result['steps_completed']) == 1
    measured = float(result['trace'][0, runtime.TRACE_FIELDS.index('reset_covariance_relative_error')])
    assert measured == pytest.approx(error, abs=2.e-6)


def test_fp64_retains_strict_covariance_guard(monkeypatch):
    result = evaluate(monkeypatch, tf.float64, .001)
    assert not bool(result['valid'])


@pytest.mark.parametrize('failure', ['nonfinite', 'covariance_nonfinite', 'rank', 'mean'])
def test_relaxed_covariance_does_not_disable_other_guards(monkeypatch, failure):
    result = evaluate(monkeypatch, tf.float32, .009, failure)
    assert not bool(result['valid'])


def test_particle_count_roundoff_allowance_and_fp32_policy():
    small = float(runtime.covariance_restoration_tolerance(32, 4, tf.float64))
    ordinary = float(runtime.covariance_restoration_tolerance(4032, 4, tf.float64))
    large = float(runtime.covariance_restoration_tolerance(100000000, 4, tf.float64))
    assert small == ordinary == pytest.approx(2.220446049250313e-16**.5)
    assert ordinary < large < .01
    for n in (32, 4032, 100000):
        tf.debugging.assert_equal(
            runtime.covariance_restoration_tolerance(n, 4, tf.float32),
            tf.constant(.01, tf.float32))
    with pytest.raises(ValueError, match="undefined"):
        runtime.covariance_restoration_tolerance(2**53, 4, tf.float64)


def test_both_consumers_use_shared_guard_and_emit_batch_predicates(monkeypatch):
    from bayesfilter.highdim import covariance_proposal_score_batch_tf as batch
    assert batch.covariance_restoration_tolerance is runtime.covariance_restoration_tolerance
    original = runtime.covariance_restoration_tolerance
    calls = []
    def checked(n, d, dtype):
        calls.append((n, d, dtype.name))
        return original(n, d, dtype)
    monkeypatch.setattr(runtime, "covariance_restoration_tolerance", checked)
    scalar = evaluate(monkeypatch, tf.float64)
    assert bool(scalar["valid"])
    monkeypatch.setattr(batch, "covariance_restoration_tolerance", checked)
    spec = LGSSMSpec("diagonal_ar", 1)
    dtype = tf.float64
    program = batch.make_batched_score_filter(spec, 32, 1,
        runtime.ProposalControls(), dtype, jit_compile=True, batch_size=2)
    result = program(tf.constant([[.9,.6,.8],[.9,.6,.8]], dtype),
        tf.constant([[.5]], dtype),
        tf.random.stateless_normal([32,1], [41,1], dtype=dtype),
        tf.random.stateless_normal([1,32,1], [41,2], dtype=dtype),
        tf.random.stateless_uniform([1,32,2], [41,3], dtype=dtype),
        tf.constant([.2,.3,.5], dtype))
    assert calls == [(32,1,"float64"), (32,1,"float64")]
    tf.debugging.assert_equal(result["validity_trace"],
                              tf.ones([2,1,len(runtime.VALIDITY_FIELDS)],tf.bool))
    tf.debugging.assert_equal(result["covariance_tolerance"],original(32,1,dtype))
    tf.debugging.assert_near(result["value"][0], scalar["value"], atol=1e-10,rtol=1e-10)
    tf.debugging.assert_near(result["score"][0,0], scalar["score"],atol=1e-9,rtol=1e-9)
