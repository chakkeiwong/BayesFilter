"""Exact shared-covariance Gaussian terms without a pair-by-state tensor."""
import tensorflow as tf


def shared_location_terms(x, means, dx, dmeans, precision):
    """Return -q/2 and its location-only tangent; covariance term is separate.

    Shapes: x/means [B,K,D], dx/dmeans [P,B,K,D], precision [B,D,D].
    Centering both primal and tangent coordinates preserves the differences.
    """
    origin = x[:, :1]
    z, m = x-origin, means-origin
    direction_origin = dx[:, :, :1]
    dz, dm = dx-direction_origin, dmeans-direction_origin
    zp = tf.linalg.matmul(z, precision)
    mp = tf.linalg.matmul(m, precision)
    distance = (tf.reduce_sum(z*zp, -1)[:, :, None]
                + tf.reduce_sum(m*mp, -1)[:, None, :]
                - 2*tf.linalg.matmul(zp, m, transpose_b=True))
    tangent = (-tf.reduce_sum(dz*zp, -1)[:, :, :, None]
               - tf.reduce_sum(dm*mp, -1)[:, :, None, :]
               + tf.linalg.matmul(dz, mp, transpose_b=True)
               + tf.linalg.matmul(zp[None], dm, transpose_b=True))
    return -.5*distance, tangent


def shared_weighted_score(weights, x, means, dx, dmeans, precision, dcovariance):
    """Average the complete Gaussian score without a [P,B,K,K] tensor.

    weights [B,I,J] are the conditional ancestor probabilities for this tile.
    precision [B,D,D] and dcovariance [P,B,D,D] are shared by those ancestors.
    All parameter directions are contracted in one matrix multiplication.
    """
    p, batch, nj, d = dmeans.shape
    origin, direction_origin = x[:, :1], dx[:, :, :1]
    z, m = x-origin, means-origin
    dz, dm = dx-direction_origin, dmeans-direction_origin
    zp = tf.linalg.matmul(z, precision)
    mp = tf.linalg.matmul(m, precision)
    h = tf.linalg.matmul(precision[None],
        tf.linalg.matmul(dcovariance, precision[None]), transpose_a=True)
    h = .5*(h+tf.linalg.matrix_transpose(h))
    hz = tf.linalg.matvec(h[:, :, None], z[None])
    hm = tf.linalg.matvec(h[:, :, None], m[None])
    m_location = tf.reduce_sum(mp[None]*dm, -1)
    m_covariance = tf.reduce_sum(m[None]*hm, -1)
    features = tf.concat([m,
        tf.reshape(tf.transpose(dm, [1, 2, 0, 3]), [batch, nj, p*d]),
        tf.transpose(m_location, [1, 2, 0]),
        tf.transpose(m_covariance, [1, 2, 0])], -1)
    averaged = tf.linalg.matmul(weights, features)
    mean_m = averaged[:, :, :d]
    mean_dm = tf.transpose(tf.reshape(averaged[:, :, d:d+p*d],
        [batch, x.shape[1], p, d]), [2, 0, 1, 3])
    mean_location = tf.transpose(averaged[:, :, d+p*d:d+p*d+p], [2, 0, 1])
    mean_covariance = tf.transpose(averaged[:, :, d+p*d+p:], [2, 0, 1])
    mean_mp = tf.linalg.matmul(mean_m, precision)
    trace = tf.einsum("bij,pbji->pb", precision, dcovariance)
    return (-tf.reduce_sum(dz*zp[None], -1)
            +tf.reduce_sum(dz*mean_mp[None], -1)
            +tf.reduce_sum(zp[None]*mean_dm, -1)-mean_location
            +.5*tf.reduce_sum(z[None]*hz, -1)
            -tf.reduce_sum(hz*mean_m[None], -1)+.5*mean_covariance
            -.5*trace[:, :, None])


def shared_gaussian_mixture(x, means, covariance, contraction="direct"):
    """Exact tiled shared-covariance density and complete weighted score."""
    import math
    from bayesfilter.highdim import ledh_small_matrix_tf as sm
    from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky
    from bayesfilter.highdim.analytical_jet_tf import Jet, logsumexp, stack
    from bayesfilter.highdim.covariance_proposal_score_tf import tile
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
    if contraction not in ("direct", "matrix"):
        raise ValueError("unknown shared Gaussian contraction")
    batch,n,d=x.v.shape; p=x.t.shape[0]; dtype=x.v.dtype
    k=select_transport_chunks(n).row_chunk_size; blocks=n//k
    covariance=covariance[:,:1]
    valid,factor=safe_cholesky(covariance.v,"importance_component")
    precision=sm.cholesky_solve(factor,tf.broadcast_to(tf.eye(d,dtype=dtype),tf.shape(factor)))[:,0]
    norm=-tf.constant(.5*d*math.log(2*math.pi)+math.log(n),dtype)-tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor[:,0])),-1)
    out=tf.TensorArray(dtype,blocks,element_shape=[batch,k])
    dout=tf.TensorArray(dtype,blocks,element_shape=[p,batch,k])
    def rows(i,out,dout):
        xv=tile(x.v,i,k,1); xt=tile(x.t,i,k,2)
        def cols(j,total,dtotal):
            mu=tile(means.v,j,k,1); dm=tile(means.t,j,k,2)
            if contraction=="matrix":
                origin=xv[:,:1]; z=xv-origin; m=mu-origin
                zp=tf.linalg.matmul(z,precision); mp=tf.linalg.matmul(m,precision)
                mahal=(tf.reduce_sum(z*zp,-1)[:,:,None]
                    +tf.reduce_sum(m*mp,-1)[:,None,:]
                    -2*tf.linalg.matmul(zp,m,transpose_b=True))
            else:
                # Sum direct coordinate differences without a [B,K,K,D] tensor.
                # Center before the linear transform to preserve translations.
                origin=xv[:,:1]
                xp=tf.linalg.matmul(xv-origin,precision)
                mp=tf.linalg.matmul(mu-origin,precision)
                def coordinate(c,total):
                    delta=xv[:,:,c,None]-mu[:,None,:,c]
                    projected=xp[:,:,c,None]-mp[:,None,:,c]
                    return c+1,total+delta*projected
                _,mahal=tf.while_loop(lambda c,*_:c<d,coordinate,
                    (0,tf.zeros([batch,k,k],dtype)),parallel_iterations=1)
            terms=norm[:,None,None]-.5*mahal
            primal=tf.reduce_logsumexp(terms,-1)
            weights=tf.nn.softmax(terms,-1)
            derivative=shared_weighted_score(weights,xv,mu,xt,dm,precision,covariance.t[:,:,0])
            block=Jet(primal,derivative)
            joined=logsumexp(stack([Jet(total,dtotal),block],-1),-1)
            return j+1,joined.v,joined.t
        _,v,t=tf.while_loop(lambda j,*_:j<blocks,cols,
            (0,tf.fill([batch,k],tf.constant(-math.inf,dtype)),tf.zeros([p,batch,k],dtype)),parallel_iterations=1)
        return i+1,out.write(i,v),dout.write(i,t)
    _,out,dout=tf.while_loop(lambda i,*_:i<blocks,rows,(0,out,dout),parallel_iterations=1)
    value=tf.reshape(tf.transpose(out.stack(),[1,0,2]),[batch,n])
    tangent=tf.reshape(tf.transpose(dout.stack(),[1,2,0,3]),[p,batch,n])
    valid=(tf.reduce_all(valid,1)&tf.reduce_all(tf.math.is_finite(value),1)
        &tf.reduce_all(tf.math.is_finite(tangent),[0,2])
        &tf.reduce_all(tf.math.is_finite(x.v),[1,2])
        &tf.reduce_all(tf.math.is_finite(means.v),[1,2]))
    return Jet(value,tangent),valid
