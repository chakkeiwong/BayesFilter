"""Marginal-law and consumer wiring checks for the actual stochastic FD path."""
import pytest
import tensorflow as tf
from bayesfilter.score_study.fd_adapter_tf import make_fd_value_kernel, fd_streams
from bayesfilter.score_study.gaussian_tf import make_particle_kernel
from bayesfilter.score_study.canonical_adapter_tf import make_canonical_kernel
from test_younis_score_master_canonical_tf import CONTROLS


def test_stream_coupling_preserves_baseline_marginal_and_independent_replicates():
    from bayesfilter.score_study.contracts import seed_pair
    row = dict(model="gaussian_all_parameters",dataset=3,replicate=2,fd_coupling="common_innovations")
    study = {"seed":991}
    common = fd_streams(row,study,0,0)
    assert common == fd_streams(row,study,6,10)
    assert common["process"] == seed_pair(master_seed=991,model=row["model"],dataset=3,
        replicate=2,stream="process",coupling_group="baseline")
    assert common != fd_streams(dict(row,replicate=3),study,0,0)
    independent = dict(row,fd_coupling="independent_nodes")
    assert fd_streams(independent,study,0,0) != fd_streams(independent,study,0,1)
    assert fd_streams(independent,study,0,0) != fd_streams(independent,study,1,0)


@pytest.mark.parametrize("proposal",["bootstrap","ledh"])
def test_value_endpoint_is_actual_consumer_with_identical_streams(proposal):
    theta = tf.constant([.62,-.8,-.6,.9,.25,-.3],tf.float64)
    obs = tf.constant([[.5],[-.2]],tf.float64)
    initial = tf.random.stateless_normal([8,2],[7,1],dtype=tf.float64)
    noise = tf.random.stateless_normal([2,8,2],[7,2],dtype=tf.float64)
    uniforms = tf.random.stateless_uniform([2,8],[7,3],dtype=tf.float64)
    design = tf.random.stateless_normal([8,2],[7,4],dtype=tf.float64)
    controls = tuple(sorted(CONTROLS.items()))
    value = make_fd_value_kernel(proposal,2,1,8,2,controls)
    if proposal == "ledh":
        actual = make_canonical_kernel(2,1,8,2,controls)(theta,tf.ones([6],tf.float64),obs,initial,noise,design)[0]
    else:
        actual = make_particle_kernel(2,1,8,2,resampling=True)(theta,obs,initial,noise,uniforms)[0]
    got = value(theta,obs,initial,noise,uniforms,design)
    tf.debugging.assert_near(got,actual,rtol=2e-12,atol=2e-12)
    assert value.experimental_get_tracing_count() == 1


def test_claim_cannot_bypass_fd_selection():
    from bayesfilter.score_study.fd_adapter_tf import evaluate_finite_difference
    with pytest.raises(ValueError,match="FD selection"):
        evaluate_finite_difference({"role":"claim"},{})
