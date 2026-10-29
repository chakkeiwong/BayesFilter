"""Independent parity and derivative checks of exact block transport."""
import pytest
import tensorflow as tf
from bayesfilter.highdim.ledh_exact_block_transport_tf import exact_block_transport_with_tangent
from bayesfilter.highdim.ledh_unified_reset_tf import batched_sinkhorn_contract_e_reset_triple_with_tangent


def fixture(n=32, dtype=tf.float64, batch=2, directions=2):
    x=tf.random.stateless_normal([batch,n,2],[42,71],dtype=dtype)
    dx=tf.random.stateless_normal([directions,batch,n,2],[13,81],dtype=dtype)*.1
    w=tf.nn.softmax(tf.random.stateless_normal([batch,n],[93,62],dtype=dtype),axis=-1)
    a=tf.random.stateless_normal([directions,batch,n],[94,62],dtype=dtype)
    dw=w[None]*(a-tf.reduce_sum(w[None]*a,-1,keepdims=True))
    cov=tf.eye(2,batch_shape=[batch,n],dtype=dtype)*(1+tf.square(x[:,:,:1]))[:,:,:,None]
    dcov=tf.eye(2,batch_shape=[directions,batch,n],dtype=dtype)*(2*x[None,:,:,:1]*dx[:,:,:,:1])[:,:,:,:,None]
    return x,dx,w,dw,cov,dcov


@pytest.mark.parametrize('regularization', [0., 1.e-7])
def test_one_block_matches_existing_reset_transport(regularization):
    x,dx,w,dw,cov,dcov=fixture()
    result=exact_block_transport_with_tangent(x,dx,w,dw,cov,dcov,epsilon=2.,iterations=12,
        denominator_regularization=regularization)
    design=x-tf.reduce_mean(x,axis=1,keepdims=True)
    original=batched_sinkhorn_contract_e_reset_triple_with_tangent(x,dx,cov,dcov,w,dw,design,
        epsilon=2.,sinkhorn_steps=8,balance_steps=4,ridge=1e-5,
        denominator_regularization=regularization)
    for a,b in ((result[4],original[4]),(result[5],original[5]),(result[2],original[2]),(result[3],original[3])):
        tf.debugging.assert_near(a,b,atol=2e-12,rtol=2e-12)
    tf.debugging.assert_near(result[0],tf.einsum('bij,bjd->bid',original[4],x),atol=2e-12,rtol=2e-12)
    tf.debugging.assert_equal(result[6],[True,True])


def test_invalid_transport_tangent_is_rejected_per_cloud():
    x,dx,w,dw,cov,dcov=fixture()
    dx=tf.concat([tf.fill(tf.shape(dx[:,:1]),tf.constant(float('nan'),dx.dtype)),dx[:,1:]],axis=1)
    result=exact_block_transport_with_tangent(x,dx,w,dw,cov,dcov,epsilon=2.,iterations=2)
    tf.debugging.assert_equal(result[6],[False,True])


def test_policy_rejects_unsupported_prime_before_tracing():
    import pytest
    x,dx,w,dw,cov,dcov=fixture(3001,batch=1,directions=1)
    with pytest.raises(ValueError,match='no divisor'):
        exact_block_transport_with_tangent(x,dx,w,dw,cov,dcov,epsilon=2.,iterations=2)
