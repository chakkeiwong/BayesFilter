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
