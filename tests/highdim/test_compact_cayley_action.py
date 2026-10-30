"""Diagnostic equivalence to dense Cayley algebra, including mixed partials."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
import numpy as np
import pytest
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import Jet,mm
from bayesfilter.highdim.covariance_proposal_moments_tf import skew_basis_factors
from bayesfilter.highdim.covariance_proposal_score_moments_tf import cayley,compact_cayley_action


@pytest.mark.parametrize('dtype',[tf.float32,tf.float64])
@pytest.mark.parametrize('nonorthogonal',[False,True])
def test_action_and_all_mixed_derivatives_match_dense(dtype,nonorthogonal):
    n,r,batch,p,d=24,4,3,3,2
    def random(shape,seed): return tf.random.stateless_normal(shape,[41,seed],dtype=dtype)
    def constant(t): return Jet(t,tf.zeros([p,*t.shape],dtype))
    u,e=skew_basis_factors(n,r,dtype)
    if nonorthogonal: u=u+random(u.shape,1)*.07
    u,e=constant(u),constant(e)
    basis=mm(mm(u,e),u.transpose())
    v=Jet(random([batch,r],2)*.3,random([p,batch,r],3)*.2)
    z=Jet(random([batch,n,d],4),random([p,batch,n,d],5))
    o,go=cayley(v,basis,.4)
    dense=(mm(o.transpose(),z),mm(go.transpose(),z[...,None,:,:]))
    compact=compact_cayley_action(v,u,e,.4,z)
    tol=1e-4 if dtype==tf.float32 else 2e-10
    for reference,actual in zip(dense,compact):
        np.testing.assert_allclose(actual.v,reference.v,atol=tol,rtol=tol)
        np.testing.assert_allclose(actual.t,reference.t,atol=tol,rtol=tol)


def test_original_cosine_basis_is_preserved():
    n,r=31,4
    u,e=skew_basis_factors(n,r,tf.float64)
    a,b=tf.transpose(u[:,::2]),tf.transpose(u[:,1::2])
    original=(a[:,:,None]*b[:,None,:]-b[:,:,None]*a[:,None,:])/tf.sqrt(tf.constant(2.,tf.float64))
    factored=tf.linalg.matmul(tf.linalg.matmul(u,e),u,transpose_b=True)
    np.testing.assert_allclose(factored,original,atol=2e-16,rtol=2e-14)


@pytest.mark.parametrize('name',['model_a','model_b'])
def test_full_filter_compact_matches_dense(name,monkeypatch):
    from bayesfilter.testing.structural_ledh_models_tf import StructuralModelSpec,make_structural_data,make_structural_design
    from bayesfilter.testing.structural_ledh_target import StructuralTarget,ready
    import bayesfilter.highdim.covariance_proposal_score_moments_tf as correction
    spec=StructuralModelSpec(name)
    data=ready(make_structural_data(spec,3,53,tf.float64));data['model']=name
    design=ready(make_structural_design(24,3,54,tf.float64))
    controls=dict(flow_steps=8,reset_steps=5,correction_steps=2)
    q=tf.stack([spec.truth(tf.float64)+i*.03 for i in range(4)])
    compact=StructuralTarget(data,design,controls,4,False,tf.float64).value_score(q)
    def dense_action(v,u,e,k,z):
        basis=mm(mm(u,e),u.transpose())
        o,go=cayley(v,basis,k)
        return mm(o.transpose(),z),mm(go.transpose(),z[...,None,:,:])
    monkeypatch.setattr(correction,'compact_cayley_action',dense_action)
    dense=StructuralTarget(data,design,controls,4,False,tf.float64).value_score(q)
    for actual,reference in zip(compact[:2],dense[:2]):
        np.testing.assert_allclose(actual,reference,atol=2e-10,rtol=2e-10)
    np.testing.assert_array_equal(compact[2],dense[2])
