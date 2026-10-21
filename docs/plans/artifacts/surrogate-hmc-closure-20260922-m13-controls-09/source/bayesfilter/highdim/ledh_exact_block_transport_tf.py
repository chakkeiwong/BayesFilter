"""Exact policy-selected blocks for the canonical finite Sinkhorn program.

All contractions use K x K blocks from the repository selector. Kernel and
tangent blocks are recomputed; the returned transport matrices are materialized
only to preserve the reset's public diagnostic contract. This is not a claim of
O(NK) total memory while that diagnostic return remains enabled.
"""
from __future__ import annotations
import tensorflow as tf
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks


def exact_block_transport_with_tangent(x, dx, weights, dw, covariance, dcovariance,
                                       *, epsilon, iterations):
    """Same primal and analytical derivatives as the one-block reset transport.

    x [B,N,D], dx [S,B,N,D]. The derivative of K r is (dK)r+K(dr),
    followed by the quotient rule for Sinkhorn scaling and actual row masses.
    Moment restoration and GenUT correction remain in their shared callers.
    """
    if x.shape[1] is None:
        raise ValueError('particle count must be static before tracing')
    selection = select_transport_chunks(int(x.shape[1]))
    n, k, blocks = selection.num_particles, selection.row_chunk_size, selection.row_blocks
    batch, directions, dim = x.shape[0], dx.shape[0], x.shape[2]
    if None in (batch, directions, dim):
        raise ValueError('batch, direction and state dimensions must be static')
    dtype=x.dtype
    nf=tf.cast(n,dtype)
    eps=tf.cast(epsilon,dtype)
    tiny=tf.cast(1e-7,dtype)

    def take(v,index,axis):
        return tf.gather(v,tf.range(k)+index*k,axis=axis)

    def geometry(i,j):
        difference=take(x,i,1)[:,:,None,:]-take(x,j,1)[:,None,:,:]
        derivative=take(dx,i,2)[:,:,:,None,:]-take(dx,j,2)[:,:,None,:,:]
        return tf.reduce_sum(difference*difference,-1),2*tf.reduce_sum(difference[None]*derivative,-1)

    def cost_body(index,total,dtotal):
        cost,dcost=geometry(index//blocks,index%blocks)
        return index+1,total+tf.reduce_sum(cost,[1,2]),dtotal+tf.reduce_sum(dcost,[2,3])
    _,cost_sum,dcost_sum=tf.while_loop(lambda index,*_:index<blocks*blocks,cost_body,
        (tf.constant(0),tf.zeros([batch],dtype),tf.zeros([directions,batch],dtype)),parallel_iterations=1)
    mean=cost_sum/(nf*nf)
    floor=tf.cast(1e-3,dtype)
    scale=tf.maximum(mean,floor)
    dscale=tf.where(mean[None]>floor,dcost_sum/(nf*nf),tf.zeros_like(dcost_sum))

    def kernel(i,j):
        cost,dcost=geometry(i,j)
        exponent=-cost/(scale[:,None,None]*eps)
        dexponent=-(dcost/scale[None,:,None,None]-cost[None]*dscale[:,:,None,None]/tf.square(scale)[None,:,None,None])/eps
        primal=tf.exp(exponent)
        return primal,primal[None]*dexponent

    def product(vector,dvector,transpose=False):
        values=tf.TensorArray(dtype,size=blocks,element_shape=[k,batch])
        derivs=tf.TensorArray(dtype,size=blocks,element_shape=[k,directions,batch])
        def row_body(i,values,derivs):
            def column_body(j,value,derivative):
                matrix,dmatrix=kernel(j,i) if transpose else kernel(i,j)
                if transpose:
                    matrix=tf.linalg.matrix_transpose(matrix)
                    dmatrix=tf.linalg.matrix_transpose(dmatrix)
                v,dv=take(vector,j,1),take(dvector,j,2)
                value+=tf.einsum('bij,bj->bi',matrix,v)
                derivative+=tf.einsum('sbij,bj->sbi',dmatrix,v)+tf.einsum('bij,sbj->sbi',matrix,dv)
                return j+1,value,derivative
            _,value,derivative=tf.while_loop(lambda j,*_:j<blocks,column_body,
                (tf.constant(0),tf.zeros([batch,k],dtype),tf.zeros([directions,batch,k],dtype)),parallel_iterations=1)
            return i+1,values.write(i,tf.transpose(value,[1,0])),derivs.write(i,tf.transpose(derivative,[2,0,1]))
        _,values,derivs=tf.while_loop(lambda i,*_:i<blocks,row_body,(tf.constant(0),values,derivs),parallel_iterations=1)
        return tf.transpose(values.concat(),[1,0]),tf.transpose(derivs.concat(),[1,2,0])

    uniform=tf.fill([batch,n],1/nf)
    def balance(index,left,right,dleft,dright):
        kr,dkr=product(right,dright)
        kr+=tiny
        left=uniform/kr
        dleft=-uniform[None]*dkr/tf.square(kr)[None]
        kl,dkl=product(left,dleft,transpose=True)
        kl+=tiny
        right=weights/kl
        dright=dw/kl[None]-weights[None]*dkl/tf.square(kl)[None]
        return index+1,left,right,dleft,dright
    _,left,right,dleft,dright=tf.while_loop(lambda index,*_:index<iterations,balance,
        (tf.constant(0),tf.ones_like(uniform),tf.ones_like(uniform),tf.zeros_like(dw),tf.zeros_like(dw)),parallel_iterations=1)

    payload=tf.concat([x,tf.reshape(covariance,[batch,n,dim*dim])],axis=-1)
    dpayload=tf.concat([dx,tf.reshape(dcovariance,[directions,batch,n,dim*dim])],axis=-1)
    width=dim+dim*dim
    values=tf.TensorArray(dtype,size=blocks,element_shape=[k,batch,width])
    derivs=tf.TensorArray(dtype,size=blocks,element_shape=[k,directions,batch,width])
    plans=tf.TensorArray(dtype,size=blocks,element_shape=[k,batch,n])
    dplans=tf.TensorArray(dtype,size=blocks,element_shape=[k,directions,batch,n])

    def row_body(i,values,derivs,plans,dplans,valid):
        l,dl=take(left,i,1),take(dleft,i,2)
        couplings=tf.TensorArray(dtype,size=blocks,element_shape=[k,batch,k])
        dcouplings=tf.TensorArray(dtype,size=blocks,element_shape=[k,directions,batch,k])
        def mass_body(j,mass,dmass,couplings,dcouplings):
            matrix,dmatrix=kernel(i,j)
            r,dr=take(right,j,1),take(dright,j,2)
            coupling=l[:,:,None]*matrix*r[:,None,:]
            dcoupling=dl[:,:,:,None]*matrix[None]*r[None,:,None,:]+l[None,:,:,None]*dmatrix*r[None,:,None,:]+l[None,:,:,None]*matrix[None]*dr[:,:,None,:]
            return j+1,mass+tf.reduce_sum(coupling,2),dmass+tf.reduce_sum(dcoupling,3),couplings.write(j,tf.transpose(coupling,[2,0,1])),dcouplings.write(j,tf.transpose(dcoupling,[3,0,1,2]))
        _,mass,dmass,couplings,dcouplings=tf.while_loop(lambda j,*_:j<blocks,mass_body,
            (tf.constant(0),tf.zeros([batch,k],dtype),tf.zeros([directions,batch,k],dtype),couplings,dcouplings),parallel_iterations=1)
        coupling=tf.transpose(couplings.concat(),[1,2,0])
        dcoupling=tf.transpose(dcouplings.concat(),[1,2,3,0])
        transport=coupling/mass[:,:,None]
        dtransport=dcoupling/mass[None,:,:,None]-coupling[None]*dmass[:,:,:,None]/tf.square(mass)[None,:,:,None]
        valid &= tf.reduce_all(tf.math.is_finite(transport),[1,2]) & tf.reduce_all(mass>0,1)
        def payload_body(j,value,derivative):
            matrix=take(transport,j,2)
            dmatrix=take(dtransport,j,3)
            p,dp=take(payload,j,1),take(dpayload,j,2)
            value+=tf.einsum('bij,bjd->bid',matrix,p)
            derivative+=tf.einsum('sbij,bjd->sbid',dmatrix,p)+tf.einsum('bij,sbjd->sbid',matrix,dp)
            return j+1,value,derivative
        _,value,derivative=tf.while_loop(lambda j,*_:j<blocks,payload_body,
            (tf.constant(0),tf.zeros([batch,k,width],dtype),tf.zeros([directions,batch,k,width],dtype)),parallel_iterations=1)
        return (i+1,values.write(i,tf.transpose(value,[1,0,2])),derivs.write(i,tf.transpose(derivative,[2,0,1,3])),
                plans.write(i,tf.transpose(transport,[1,0,2])),dplans.write(i,tf.transpose(dtransport,[2,0,1,3])),valid)
    _,values,derivs,plans,dplans,valid=tf.while_loop(lambda i,*_:i<blocks,row_body,
        (tf.constant(0),values,derivs,plans,dplans,tf.ones([batch],tf.bool)),parallel_iterations=1)
    value=tf.transpose(values.concat(),[1,0,2])
    derivative=tf.transpose(derivs.concat(),[1,2,0,3])
    return (value[:,:,:dim],derivative[:,:,:,:dim],
            tf.reshape(value[:,:,dim:],[batch,n,dim,dim]),
            tf.reshape(derivative[:,:,:,dim:],[directions,batch,n,dim,dim]),
            tf.transpose(plans.concat(),[1,0,2]),tf.transpose(dplans.concat(),[1,2,0,3]),valid)
