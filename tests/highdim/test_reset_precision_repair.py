"""CPU-only reset checks: mixed rows, tangents, unchanged healthy path, guards."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import Jet
from bayesfilter.highdim.covariance_proposal_tf import ProposalControls
from bayesfilter.highdim import covariance_proposal_score_batch_tf as batch


def test_mixed_row_precision_repair_and_analytical_tangents(monkeypatch):
    # Identity transport isolates moments/recolouring, including a truly singular row.
    def identity_transport(x,logw,controls,precision_policy):
        return x,tf.zeros([3],x.v.dtype),tf.zeros([3],x.v.dtype)
    monkeypatch.setattr(batch,'precision_transport',identity_transport)
    base=np.array([[-1,-1],[-1,1],[1,-1],[1,1],[-2,0],[2,0],[0,-2],[0,2]],float)
    values=np.stack([base,base*[1,1e-4],base*[1,0]])
    tangents=np.stack([values*.13,values*[-.03,.17]])
    lw=np.broadcast_to(np.log(np.arange(1,9)/36),(3,8)).copy()
    lt=np.stack([np.broadcast_to(np.linspace(-.1,.1,8),(3,8)),np.zeros((3,8))])
    controls=ProposalControls()
    def run(dtype,policy):
        x=Jet(tf.constant(values,dtype),tf.constant(tangents,dtype))
        w=Jet(tf.constant(lw,dtype),tf.constant(lt,dtype))
        @tf.function(input_signature=[],autograph=False)
        def graph():
            z,mu,p,valid,rd,used=batch.reset_cloud(x,w,controls,reset_precision_policy=policy,return_fallback=True)
            return z.v,z.t,valid,used
        return graph()
    original=run(tf.float32,'uniform');repaired=run(tf.float32,'conditional_fp64');ref=run(tf.float64,'uniform')
    np.testing.assert_array_equal(original[2],[True,False,False])
    np.testing.assert_array_equal(repaired[2],[True,True,False])
    np.testing.assert_array_equal(repaired[3],[False,True,True])
    np.testing.assert_array_equal(repaired[0][0],original[0][0])
    np.testing.assert_array_equal(repaired[1][:,0],original[1][:,0])
    np.testing.assert_allclose(repaired[0][1],ref[0][1],rtol=3e-6,atol=1e-7)
    np.testing.assert_allclose(repaired[1][:,1],ref[1][:,1],rtol=5e-5,atol=1e-7)


def test_policy_is_opt_in_and_target_identity_changes():
    from bayesfilter.testing.structural_ledh_models_tf import StructuralModelSpec,make_structural_data,make_structural_design
    from bayesfilter.testing.structural_ledh_target import StructuralTarget,ready
    spec=StructuralModelSpec('model_b')
    data=ready(make_structural_data(spec,2,61));data['model']='model_b'
    design=ready(make_structural_design(16,2,62))
    controls=dict(flow_steps=8,reset_steps=5,correction_steps=1)
    original=StructuralTarget(data,design,controls,4,False)
    explicit=StructuralTarget(data,design,controls,4,False,reset_precision_policy='uniform')
    repaired=StructuralTarget(data,design,controls,4,False,reset_precision_policy='conditional_fp64')
    assert original.target_signature==explicit.target_signature
    assert original.target_signature!=repaired.target_signature
    q=tf.stack([spec.truth()+i*.01 for i in range(4)])
    a=original.program(q,original.observations,original.design['initial'],original.design['innovations'])
    b=repaired.program(q,repaired.observations,repaired.design['initial'],repaired.design['innovations'])
    assert not bool(tf.reduce_any(b['reset_precision_fallback']))
    for key in ('value','score','valid'):
        np.testing.assert_array_equal(a[key],b[key])
    # Separate CPU graphs can reorder the downstream moment-correction algebra.
    # Reset-row bitwise preservation is tested directly above; full recurrence
    # tangents are checked numerically, rather than asserting compiler identity.
    for key in ('particle_tangent','particles'):
        np.testing.assert_allclose(a[key],b[key],rtol=1e-3,atol=3e-5)
