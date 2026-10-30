"""CPU-only XLA boundary regression for the fixed proposal design."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
from fractions import Fraction
import pytest
import tensorflow as tf
from bayesfilter.highdim import covariance_proposal_tf as directional
from bayesfilter.highdim import covariance_proposal_score_tf as recursive
from bayesfilter.highdim.analytical_jet_tf import Jet


def oracle(beta, uniforms, n):
    thresholds = []
    total = Fraction(0)
    for value in beta[:-1]:
        total += Fraction(value)
        thresholds.append(total)
    components = [sum(Fraction(u[0]) >= t for t in thresholds) for u in uniforms]
    ancestors = [min(int(Fraction(u[1])*n), n-1) for u in uniforms]
    return components, ancestors


def test_identical_indices_at_real_4032_ancestor_and_mixture_boundaries():
    # First three ancestor coordinates are the observed SIR failures.
    inputs = tf.constant([[.05,.2819940447807312], [.4,.3132440447807312],
        [.9,.4551091194152832], [.1,0.], [.3,.5], [1.,1.]], tf.float32)
    beta = tf.constant([.1,.2,.7], tf.float32)
    expected = oracle(beta.numpy().tolist(), inputs.numpy().tolist(), 4032)
    for dtype in (tf.float32, tf.float64):
        kernel = tf.function(lambda b,u: directional.fixed_design_indices(b,u,4032),
            input_signature=[tf.TensorSpec([3],dtype),tf.TensorSpec([6,2],dtype)], jit_compile=True)
        actual = kernel(tf.cast(beta,dtype),tf.cast(inputs,dtype))
        assert [v.numpy().tolist() for v in actual] == list(expected)
    legacy = tf.cast(inputs[:3,1]*4032,tf.int32).numpy().tolist()
    assert legacy == [1137,1263,1835]
    assert expected[1][:3] == [1136,1262,1834]


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_actual_directional_and_all_parameter_draws_share_fixed_design(dtype):
    n=4032
    sample=tf.constant([[.05,.2819940447807312], [.4,.3132440447807312],
                        [.9,.4551091194152832]],tf.float32)
    uniforms=tf.cast(tf.tile(sample,[n//3,1]),dtype)
    beta32=tf.constant([.1,.2,.7],tf.float32)
    beta=tf.cast(beta32,dtype)
    b,j=oracle(beta32.numpy().tolist(),tf.cast(uniforms,tf.float32).numpy().tolist(),n)
    expected_value=tf.constant([[a+10000*c] for c,a in zip(b,j)],dtype)
    expected_tangent=tf.constant([[2*a+3+c] for c,a in zip(b,j)],dtype)
    @tf.function(input_signature=[tf.TensorSpec([n,2],dtype),tf.TensorSpec([3],dtype)],jit_compile=True)
    def draw(u,weights):
        m=tf.cast(tf.range(n)[:,None],dtype)
        dm=2*m+3
        a=tf.ones([3,n,1,1],dtype); da=tf.zeros_like(a)
        c=tf.broadcast_to(tf.cast(tf.range(3)[:,None,None],dtype)*10000,[3,n,1])
        dc=tf.broadcast_to(tf.cast(tf.range(3)[:,None,None],dtype),[3,n,1])
        l=tf.ones([1,1],dtype); dl=tf.zeros_like(l)
        normals=tf.zeros([n,1],dtype)
        maps=(None,None,None,None,a,da,c,dc,m,dm,l,dl)
        v,d=directional.draw_from_maps(maps,weights,u,normals)
        maps2=(None,None,Jet(a,da[None]),Jet(c,dc[None]),Jet(m,dm[None]),Jet(l,dl[None]))
        jet=recursive.draw_from_maps(maps2,weights,u,normals)
        prepared=recursive.draw_from_maps(maps2,weights,u,normals,
            indices=(tf.constant(b,tf.int32),tf.constant(j,tf.int32)))
        return v,d,jet.v,jet.t[0],prepared.v,prepared.t[0]
    value,tangent,jet_value,jet_tangent,prepared_value,prepared_tangent=draw(uniforms,beta)
    tf.debugging.assert_equal(value,expected_value)
    tf.debugging.assert_equal(tangent,expected_tangent)
    tf.debugging.assert_equal(jet_value,expected_value)
    tf.debugging.assert_equal(jet_tangent,expected_tangent)
    tf.debugging.assert_equal(prepared_value,expected_value)
    tf.debugging.assert_equal(prepared_tangent,expected_tangent)


def test_bound_sir_design_matches_dynamic_filter_and_compiles_without_fp64():
    """Real SIR recurrence parity; tiny CPU-only XLA diagnostic, not admission."""
    import re
    from docs.benchmarks.run_ledh_covariance_proposal import spec_for
    from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
    spec = spec_for("sir_d18", lgssm_family="p44")
    n, horizon, batch = 32, 2, 2
    beta = tf.constant([.1, .06849747, .83150253], tf.float32)
    uniforms = tf.random.stateless_uniform([horizon,n,2], [20261020,21])
    inputs = (tf.zeros([batch,3]), tf.fill([horizon,9],tf.constant(20.)),
              tf.random.stateless_normal([n,18],[20261020,22]),
              tf.random.stateless_normal([horizon,n,18],[20261020,23]))
    controls = directional.ProposalControls(flow_steps=16,reset_steps=5,
        reset_epsilon=1.,correction_steps=0)
    dynamic = make_batched_score_filter(spec,n,horizon,controls,tf.float32,
                                      batch_size=batch,jit_compile=True)
    bound = make_batched_score_filter(spec,n,horizon,controls,tf.float32,
        batch_size=batch,jit_compile=True,fixed_draw_design=(beta,uniforms))
    original = dynamic(*inputs,uniforms,beta)
    prepared = bound(*inputs)
    for key in ("value","score","particles","particle_tangent","valid","steps_completed"):
        tf.debugging.assert_equal(original[key],prepared[key],message=key)
    hlo = bound.experimental_get_compiler_ir(*inputs)(stage="optimized_hlo")
    assert not re.search(r"\bf64\[",hlo), "FP64 survived XLA optimization"
    assert len(bound.input_signature) == 4  # Frozen beta/uniforms cannot be overridden.
    with pytest.raises((TypeError,ValueError)):
        bound(*inputs,uniforms,beta)
