"""Independent reference checks; autodiff is diagnostic only."""
import pytest
import tensorflow as tf
from bayesfilter.highdim.ledh_marginal_weights_tf import gaussian_mixture_log_density_tangent, marginal_prior_ratio_tangent
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import _single_cloud_model
from bayesfilter.highdim.ledh_canonical_score_tf import _flow_substeps_with_tangent
from bayesfilter.testing.marginal_weight_campaign import specification

def inputs(d=3, n=5):
    dtype = tf.float64
    raw = tf.random.stateless_normal([n,d,d], [21,12], dtype=dtype)
    c = raw@tf.linalg.matrix_transpose(raw)+tf.eye(d,batch_shape=[n],dtype=dtype)
    dc = tf.random.stateless_normal([n,d,d], [21,13], dtype=dtype)
    dc = .02*(dc+tf.linalg.matrix_transpose(dc))
    x = tf.random.stateless_normal([n,d], [21,14], dtype=dtype)
    m = tf.random.stateless_normal([n,d], [21,15], dtype=dtype)
    dx = tf.random.stateless_normal([n,d], [21,16], dtype=dtype)*.1
    dm = tf.random.stateless_normal([n,d], [21,17], dtype=dtype)*.1
    lw = tf.nn.log_softmax(tf.linspace(tf.constant(-1.,dtype),tf.constant(1.,dtype),n))
    dlw = tf.linspace(tf.constant(-.2,dtype),tf.constant(.3,dtype),n)
    dlw -= tf.reduce_sum(tf.exp(lw)*dlw)
    return x,dx,m,dm,c,dc,lw,dlw

@pytest.mark.parametrize("d", [1,3])
def test_nonuniform_mixture_total_derivative(d):
    x,dx,m,dm,c,dc,lw,dlw = inputs(d)
    value,derivative,vv,fv = gaussian_mixture_log_density_tangent(x,dx,m,dm,c,dc,lw,dlw)
    assert bool(vv & fv)
    import tensorflow_probability as tfp
    a = tf.constant(0.,tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(a)
        terms = tfp.distributions.MultivariateNormalTriL(m+a*dm,tf.linalg.cholesky(c+a*dc)).log_prob((x+a*dx)[:,None])
        ref = tf.reduce_logsumexp(terms+lw+a*dlw,axis=1)
        scalar = tf.reduce_sum(ref)
    tf.debugging.assert_near(value,ref,atol=1e-12,rtol=1e-12)
    tf.debugging.assert_near(tf.reduce_sum(derivative),tape.gradient(scalar,a),atol=1e-11,rtol=1e-11)
    for h in (1e-4,3e-5):
        plus = gaussian_mixture_log_density_tangent(x+h*dx,dx,m+h*dm,dm,c+h*dc,dc,lw+h*dlw,dlw)[0]
        minus = gaussian_mixture_log_density_tangent(x-h*dx,dx,m-h*dm,dm,c-h*dc,dc,lw-h*dlw,dlw)[0]
        tf.debugging.assert_near(derivative,(plus-minus)/(2*h),atol=1e-8,rtol=1e-8)

def test_identity_flow_cancels_mixture_and_tangent():
    x,dx,m,dm,c,dc,lw,dlw = inputs()
    b = tf.eye(3,batch_shape=[5],dtype=tf.float64)
    result = marginal_prior_ratio_tangent(x,dx,m,dm,c[0],dc[0],m,dm,b,tf.zeros_like(b),lw,dlw)
    tf.debugging.assert_near(result[0],tf.zeros([5],tf.float64),atol=1e-13)
    tf.debugging.assert_near(result[1],tf.zeros([5],tf.float64),atol=1e-13)
    assert bool(result[2] & result[3])

def test_invalid_component_is_not_silently_ridged():
    x,dx,m,dm,c,dc,lw,dlw = inputs()
    c = tf.tensor_scatter_nd_update(c,[[0,0,0]],[tf.constant(-100.,tf.float64)])
    assert not bool(gaussian_mixture_log_density_tangent(x,dx,m,dm,c,dc,lw,dlw)[2])

def test_affine_bookkeeping_matches_actual_nonlinear_flow():
    spec = specification("m13-t1",tf.float64)
    theta,direction = tf.constant([-.3,.2],tf.float64),tf.constant([.3,-.2],tf.float64)
    model = _single_cloud_model(spec["model"],theta,direction)
    anchors = tf.constant([[1.2,.7,.18,-.06],[1.,.8,.15,.03]],tf.float64)
    da = tf.constant([[.1,-.1,.02,.03],[-.1,.04,.03,-.01]],tf.float64)
    innovation = tf.constant([[.03,-.02,.01,.02],[-.02,.03,-.01,.01]],tf.float64)
    di = .1*innovation
    p = tf.broadcast_to(tf.constant(.04,tf.float64)*tf.eye(4,dtype=tf.float64),[2,4,4]);dp=.02*p
    r = model.observation_covariance; dr = model.observation_covariance_tangent_fn(theta);rinv=tf.linalg.inv(r)
    out = _flow_substeps_with_tangent(model,anchors+innovation,da+di,anchors,da,p,dp,
        spec["fixture"].observations[0],r,dr,rinv,-rinv@dr@rinv,substeps=10,eye=tf.eye(4,dtype=tf.float64),return_affine=True)
    x,dx,_,_,vv,fv,mean,dmean,b,db = out
    tf.debugging.assert_equal(tf.reduce_all(vv & fv),True)
    tf.debugging.assert_near(x,mean+tf.linalg.matvec(b,innovation),atol=1e-12,rtol=1e-12)
    tf.debugging.assert_near(dx,dmean+tf.linalg.matvec(db,innovation)+tf.linalg.matvec(b,di),atol=1e-12,rtol=1e-12)

    # The Gaussian pushforward additionally requires that its coefficients
    # do not depend on this child's fresh innovation. Affine bookkeeping alone
    # would not detect coefficients evaluated on the actual random child.
    changed_innovation = -2.3*innovation
    changed_di = 1.7*di
    other = _flow_substeps_with_tangent(
        model,anchors+changed_innovation,da+changed_di,anchors,da,p,dp,
        spec["fixture"].observations[0],r,dr,rinv,-rinv@dr@rinv,
        substeps=10,eye=tf.eye(4,dtype=tf.float64),return_affine=True)
    tf.debugging.assert_equal(tf.reduce_all(other[4] & other[5]),True)
    for original_coefficient,changed_coefficient in zip(out[6:],other[6:]):
        tf.debugging.assert_near(original_coefficient,changed_coefficient,atol=1e-12,rtol=1e-12)
    tf.debugging.assert_near(other[0],mean+tf.linalg.matvec(b,changed_innovation),atol=1e-12,rtol=1e-12)
    tf.debugging.assert_near(other[1],
        dmean+tf.linalg.matvec(db,changed_innovation)+tf.linalg.matvec(b,changed_di),
        atol=1e-12,rtol=1e-12)



@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_precision_policy_keeps_invalid_covariance_invalid(dtype):
    args=[tf.cast(x,dtype) for x in inputs()]
    x,dx,m,dm,c,dc,lw,dlw=args
    q=tf.linalg.diag(tf.constant([1.,1.,-1.],dtype))
    b=tf.eye(3,batch_shape=[5],dtype=dtype)
    result=marginal_prior_ratio_tangent(x,dx,m,dm,q,dc[0],m,dm,b,
        tf.zeros_like(b),lw,dlw,return_precision_diagnostics=True)
    assert not bool(result[2])
    assert int(result[4])==0


def test_fp64_precision_observability_preserves_identity_cancellation():
    x,dx,m,dm,c,dc,lw,dlw=inputs()
    b=tf.eye(3,batch_shape=[5],dtype=tf.float64)
    result=marginal_prior_ratio_tangent(x,dx,m,dm,c[0],dc[0],m,dm,b,
        tf.zeros_like(b),lw,dlw,return_precision_diagnostics=True)
    assert int(result[4])==0 and bool(result[2] & result[3])
    tf.debugging.assert_near(result[0],tf.zeros([5],tf.float64),atol=1e-13)
    tf.debugging.assert_near(result[1],tf.zeros([5],tf.float64),atol=1e-13)
