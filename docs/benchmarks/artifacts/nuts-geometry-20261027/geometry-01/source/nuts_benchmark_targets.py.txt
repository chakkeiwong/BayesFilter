"""Analytic synthetic targets for the standalone XLA sampler diagnostic.

Every reference coordinate z is standard normal. The inverse transforms and
Jacobian terms below define the densities; no filtering-model claim is made.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    name: str
    dimension: int
    log_prob: object
    from_reference: object
    to_reference: object
    description: str


NAMES = ('gaussian32', 'rotated_gaussian32', 'banana8', 'funnel10',
         'noncentered_funnel10', 'regression32')


def make_target(tf, name):
    if name not in (*NAMES, 'banana2'):
        raise ValueError(f'Unknown target: {name}')
    dimension = 2 if name == 'banana2' else (8 if name == 'banana8' else
                 (10 if 'funnel' in name else 32))
    identity = lambda q: q
    normal = lambda q: -.5 * tf.reduce_sum(tf.square(q), axis=-1)
    if name in ('gaussian32', 'noncentered_funnel10'):
        return Target(name, dimension, normal, identity, identity,
                      'Independent standard normal; exact noncentered control'
                      if 'funnel' in name else 'Isotropic Gaussian control')
    if name.startswith('banana'):
        bend = tf.constant(.5 if dimension == 2 else 1., tf.float64)
        def transform(q, sign):
            pairs = tf.reshape(q, [-1, dimension // 2, 2])
            x = pairs[..., 0]
            y = pairs[..., 1] + sign * bend * (tf.square(x) - 1.)
            return tf.reshape(tf.stack((x, y), axis=-1), tf.shape(q))
        inverse = lambda q: transform(q, -1.)
        return Target(name, dimension, lambda q: normal(inverse(q)),
                      lambda z: transform(z, 1.), inverse,
                      'Unit-Jacobian product banana; bend .5 (d2) or 1 (d8)')
    if name == 'rotated_gaussian32':
        scales = tf.exp(tf.linspace(tf.constant(0., tf.float64),
                                   tf.math.log(tf.constant(1000., tf.float64)) / 2., dimension))
        unit = tf.ones([dimension], tf.float64) / tf.sqrt(tf.cast(dimension, tf.float64))
        def rotate(q):
            return q - 2. * tf.reduce_sum(q * unit, axis=-1, keepdims=True) * unit
        inverse = lambda q: rotate(q) / scales
        return Target(name, dimension, lambda q: normal(inverse(q)),
                      lambda z: rotate(z * scales), inverse,
                      'Householder-rotated Gaussian; covariance eigenvalues 1..1000')
    if name == 'funnel10':
        def forward(z):
            v = 3. * z[..., :1]
            return tf.concat((v, tf.exp(v / 2.) * z[..., 1:]), axis=-1)
        def inverse(q):
            v = q[..., :1]
            return tf.concat((v / 3., q[..., 1:] * tf.exp(-v / 2.)), axis=-1)
        def density(q):
            # v~N(0,9); x|v~N(0,exp(v) I). The -(d-1)v/2 term
            # is required by the conditional normal normalization.
            return normal(inverse(q)) - (dimension - 1.) * q[..., 0] / 2.
        return Target(name, dimension, density, forward, inverse,
                      'Centered funnel; v standard deviation 3, nine conditional coordinates')
    # A deterministic full-rank Gaussian regression posterior. The target
    # deliberately evaluates X beta, not the equivalent collapsed quadratic.
    design = tf.random.stateless_normal([512, dimension], [20261027, 61], dtype=tf.float64)
    precision = tf.eye(dimension, dtype=tf.float64) + tf.matmul(design, design, transpose_a=True) / 512.
    chol = tf.linalg.cholesky(precision)
    def forward(z):
        return tf.transpose(tf.linalg.triangular_solve(chol, tf.transpose(z), adjoint=True))
    inverse = lambda q: tf.matmul(q, chol)
    def density(q):
        residual = tf.matmul(q, design, transpose_b=True)
        return normal(q) - .5 * tf.reduce_sum(tf.square(residual), axis=-1) / 512.
    return Target(name, dimension, density, forward, inverse,
                  'Gaussian regression: 512x32 design, zero response, unit normal prior')
