"""CPU-only independent dense checks of shared Gaussian contractions."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import pytest
import tensorflow as tf
from bayesfilter.highdim.covariance_proposal_shared_gaussian_tf import shared_location_terms

@pytest.mark.parametrize("dimension", [1, 3, 18])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("offset,spread", [(0., 1.), (10000., 1.), (100., .01)])
def test_centered_shared_gaussian_matches_dense_residual(dimension, dtype, offset, spread):
    p,b,n,d=4,2,9,dimension
    x=tf.cast(offset,dtype)+tf.cast(spread,dtype)*tf.random.stateless_normal([b,n,d],[912,1],dtype=dtype)
    means=tf.cast(offset,dtype)+tf.cast(spread,dtype)*tf.random.stateless_normal([b,n,d],[912,2],dtype=dtype)
    dx=tf.cast(offset,dtype)+tf.random.stateless_normal([p,b,n,d],[912,3],dtype=dtype)
    dm=tf.cast(offset,dtype)+tf.random.stateless_normal([p,b,n,d],[912,4],dtype=dtype)
    factor=tf.random.stateless_normal([b,d,d],[912,5],dtype=dtype)
    precision=tf.linalg.matmul(factor,factor,transpose_b=True)+tf.eye(d,batch_shape=[b],dtype=dtype)
    actual_value,actual_tangent=shared_location_terms(x,means,dx,dm,precision)
    # Dense reference evaluates the same quantized inputs in double precision.
    xx,mm,ddx,ddm,pr=[tf.cast(z,tf.float64) for z in (x,means,dx,dm,precision)]
    residual=xx[:,:,None]-mm[:,None]
    solved=tf.linalg.matvec(pr[:,None,None],residual)
    expected_value=-.5*tf.reduce_sum(residual*solved,-1)
    expected_tangent=-tf.reduce_sum(solved[None]*(ddx[:,:,:,None]-ddm[:,:,None]),-1)
    tolerance=4e-5 if dtype==tf.float32 else 2e-11
    for actual,expected in ((actual_value,expected_value),(actual_tangent,expected_tangent)):
        scale=tf.maximum(tf.constant(1.,tf.float64),tf.reduce_max(tf.abs(expected)))
        tf.debugging.assert_less_equal(tf.reduce_max(tf.abs(tf.cast(actual,tf.float64)-expected))/scale,tf.constant(tolerance,tf.float64))

def test_identical_points_and_sensitivities_cancel():
    x=tf.constant([[[100.,101.],[102.,103.]]],tf.float64)
    dx=tf.constant([[[[1000.,1001.],[1002.,1003.]]]],tf.float64)
    value,tangent=shared_location_terms(x,x,dx,dx,tf.eye(2,batch_shape=[1],dtype=tf.float64))
    tf.debugging.assert_equal(tf.linalg.diag_part(value),tf.zeros([1,2],tf.float64))
    tf.debugging.assert_equal(tf.linalg.diag_part(tangent),tf.zeros([1,1,2],tf.float64))


@pytest.mark.parametrize("dimension", [1, 3, 18])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("offset", [0., 10000.])
def test_weighted_total_score_matches_independent_dense_formula(dimension, dtype, offset):
    from bayesfilter.highdim.covariance_proposal_shared_gaussian_tf import shared_weighted_score
    p,b,i,j,d=3,2,7,9,dimension
    def noise(shape, seed):
        return tf.random.stateless_normal(shape, [745, seed], dtype=dtype)
    x= tf.cast(offset,dtype)+noise([b,i,d],1)
    m= tf.cast(offset,dtype)+noise([b,j,d],2)
    dx=tf.cast(offset,dtype)+noise([p,b,i,d],3)
    dm=tf.cast(offset,dtype)+noise([p,b,j,d],4)
    factor=noise([b,d,d],5)
    pr=tf.linalg.matmul(factor,factor,transpose_b=True)+tf.eye(d,batch_shape=[b],dtype=dtype)
    dc=noise([p,b,d,d],6)
    dc=.5*(dc+tf.linalg.matrix_transpose(dc))
    w=tf.nn.softmax(noise([b,i,j],7),-1)
    signature=[tf.TensorSpec(t.shape,t.dtype) for t in (w,x,m,dx,dm,pr,dc)]
    function=tf.function(shared_weighted_score,input_signature=signature,jit_compile=True)
    actual=function(w,x,m,dx,dm,pr,dc)
    ww,xx,mm,ddx,ddm,pp,cc=[tf.cast(v,tf.float64) for v in (w,x,m,dx,dm,pr,dc)]
    residual=xx[:,:,None]-mm[:,None]
    u=tf.linalg.matvec(pp[:,None,None],residual)
    location=-tf.reduce_sum(u[None]*(ddx[:,:,:,None]-ddm[:,:,None]),-1)
    quadratic=tf.einsum("bijd,pbdk,bijk->pbij",u,cc,u)
    trace=tf.einsum("bij,pbji->pb",pp,cc)
    expected=tf.reduce_sum(ww[None]*(location+.5*quadratic-.5*trace[:,:,None,None]),-1)
    scale=tf.maximum(tf.constant(1.,tf.float64),tf.reduce_max(tf.abs(expected)))
    tf.debugging.assert_less_equal(tf.reduce_max(tf.abs(tf.cast(actual,tf.float64)-expected))/scale,
        tf.constant(2e-4 if dtype==tf.float32 else 2e-11,tf.float64))
    assert function.experimental_get_tracing_count()==1
