"""CPU-only exact-tile primitive checks; small fixtures obey K=N."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from bayesfilter.testing.surrogate_hmc_transport_tile_batch_diagnostic import _inputs,serial_reduce,batched_reduce


def test_one_tile_fp64_value_and_analytical_score():
    args=_inputs(7,7,2,2,3,tf.float64)
    for rows in (True,False):
        baseline=serial_reduce(*args,7,rows)
        candidate=batched_reduce(*args,7,rows)
        for a,b in zip(candidate,baseline):
            tf.debugging.assert_near(a,b,atol=1e-11,rtol=1e-11)
        x,t,p,pt,_=args
        h=tf.constant(1e-5,tf.float64)
        plus=_inputs_for_perturb(x+h*t[0],p+h*pt[0])
        minus=_inputs_for_perturb(x-h*t[0],p-h*pt[0])
        finite=(serial_reduce(*plus,7,rows)[0]-serial_reduce(*minus,7,rows)[0])/(2*h)
        tf.debugging.assert_near(candidate[1][0],finite,atol=2e-8,rtol=2e-8)


def _inputs_for_perturb(x,p):
    b,n,d=x.shape
    delta=x[:,:,None,:]-x[:,None,:,:]
    c=-tf.reduce_sum(delta*delta,-1)/tf.cast(max(1,d),x.dtype)
    return x,tf.zeros([1,b,n,d],x.dtype),p,tf.zeros([1,b,n],x.dtype),c
