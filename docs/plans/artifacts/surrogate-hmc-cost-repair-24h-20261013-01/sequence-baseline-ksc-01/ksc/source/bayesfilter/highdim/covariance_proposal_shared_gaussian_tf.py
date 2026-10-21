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
