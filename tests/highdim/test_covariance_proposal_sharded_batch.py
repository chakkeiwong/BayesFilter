"""CPU/XLA diagnostic parity of logical batches and native batched shards."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import pytest
import tensorflow as tf
from bayesfilter.highdim.covariance_proposal_score_sharded_tf import make_sharded_score_filter
from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
from bayesfilter.highdim.covariance_proposal_tf import ProposalControls
from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec


@pytest.mark.parametrize("fixed,history", [(True, False), (False, True)])
def test_shards_preserve_every_output_and_padded_row_isolation(fixed, history):
    dtype = tf.float64
    spec = NonlinearSQMCSpec("sir_d18")
    n, horizon, batch = 32, 2, 5
    controls = ProposalControls(flow_steps=2, correction_steps=2)
    initial = tf.random.stateless_normal([n,18], [813,1], dtype=dtype)
    noise = tf.random.stateless_normal([horizon,n,18], [813,2], dtype=dtype)
    uniforms = tf.random.stateless_uniform([horizon,n,2], [813,3], dtype=dtype)
    beta = tf.constant([.2,.3,.5], dtype)
    observations = spec.observations(horizon, 813, dtype=dtype)
    theta = tf.constant([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.],[0.,0.,.01],[float("nan"),0.,0.]], dtype)
    options = dict(dtype=dtype, jit_compile=True, batch_size=batch,
                   record_history=history, fixed_draw_design=(beta,uniforms) if fixed else None)
    full = make_batched_score_filter(spec,n,horizon,controls,**options)
    sharded = make_sharded_score_filter(spec,n,horizon,controls,shard_size=2,**options)
    args = (theta,observations,initial,noise) + (() if fixed else (uniforms,beta))
    a,b = full(*args),sharded(*args)
    assert set(a) == set(b)
    for key in a:
        if a[key].dtype.is_floating:
            tf.debugging.assert_equal(tf.math.is_finite(a[key]),tf.math.is_finite(b[key]))
            clean = lambda value: tf.where(tf.math.is_finite(value),value,tf.zeros_like(value))
            tf.debugging.assert_near(clean(a[key]),clean(b[key]),atol=1e-8,rtol=1e-8)
        else:
            tf.debugging.assert_equal(a[key],b[key])
    tf.debugging.assert_equal(a["valid"],[True,True,True,True,False])


def test_scalar_shards_are_forbidden():
    with pytest.raises(ValueError,match="shard_size"):
        make_sharded_score_filter(NonlinearSQMCSpec("sir_d18"),32,2,
            ProposalControls(),batch_size=4,shard_size=1)
