"""CPU-only algebraic transport equivalence; no model or posterior claim."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
os.environ.setdefault('BAYESFILTER_PRELOAD_CUSTOM_OP','0')
import pytest
import tensorflow as tf
from bayesfilter.highdim import covariance_proposal_score_batch_tf as shared
from bayesfilter.highdim.analytical_jet_tf import Jet,logsumexp
from bayesfilter.highdim.covariance_proposal_tf import ProposalControls
from bayesfilter.testing import surrogate_hmc_transport_arithmetic_diagnostic as diagnostic


@pytest.mark.parametrize('kind',['gauge','direct','direct_gauge'])
def test_transport_algebraic_equivalence(tmp_path,kind):
    n,d,p,b=16,3,2,2;dt=tf.float64
    controls=ProposalControls(reset_steps=4,reset_epsilon=2.)
    x=tf.random.stateless_normal([b,n,d],[7,1],dtype=dt)
    xt=tf.random.stateless_normal([p,b,n,d],[7,2],dtype=dt)*3
    w=Jet(tf.random.stateless_normal([b,n],[7,3],dtype=dt),tf.random.stateless_normal([p,b,n],[7,4],dtype=dt))
    w=w-logsumexp(w,-1,True)
    args=(x,xt,w.v,w.t)
    fn,digest=diagnostic.load_candidate(tmp_path/(kind+'.py'),kind)
    ref=diagnostic.graph(shared.transport,b,n,d,p,dt,controls)(*args)
    out=diagnostic.graph(fn,b,n,d,p,dt,controls)(*args)
    assert len(digest)==64
    for a,r in zip(out,ref):tf.debugging.assert_near(a,r,atol=1e-9,rtol=1e-9)
    with pytest.raises(ValueError):
        diagnostic.candidate_source('invalid source',kind)
