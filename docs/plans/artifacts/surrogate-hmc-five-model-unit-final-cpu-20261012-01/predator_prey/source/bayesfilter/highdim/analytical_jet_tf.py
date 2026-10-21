"""Explicit first-order analytical algebra with one leading sensitivity axis.

This is ordinary TensorFlow tensor algebra, not an autodiff/JVP engine. The
primal is stored once; derivative blocks have shape [P, *primal.shape].
"""
import tensorflow as tf


class Jet:
    def __init__(self, value, tangent):
        self.v = value
        self.t = tangent

    def constant(self, value):
        value = tf.convert_to_tensor(value, dtype=self.v.dtype)
        return Jet(value, tf.zeros(tf.concat([[self.t.shape[0]],tf.shape(value)],0), value.dtype))

    def coerce(self, other):
        return other if isinstance(other, Jet) else self.constant(other)

    def _aligned(self, rank):
        if rank == self.v.shape.rank:
            return self.t
        return tf.reshape(self.t, tf.concat([[self.t.shape[0]] + [1]*(rank-self.v.shape.rank), tf.shape(self.v)],0))

    def __add__(self, other):
        other = self.coerce(other)
        value = self.v + other.v
        return Jet(value, self._aligned(value.shape.rank) + other._aligned(value.shape.rank))

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.v, -self.t)

    def __sub__(self, other):
        return self + -self.coerce(other)

    def __rsub__(self, other):
        return self.coerce(other) + -self

    def __mul__(self, other):
        other = self.coerce(other)
        value = self.v * other.v
        return Jet(value, self._aligned(value.shape.rank)*other.v + self.v*other._aligned(value.shape.rank))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.coerce(other)
        value = self.v / other.v
        return Jet(value, (self._aligned(value.shape.rank)-value*other._aligned(value.shape.rank))/other.v)

    def __rtruediv__(self, other):
        return self.coerce(other) / self

    def __pow__(self, exponent):
        return Jet(self.v**exponent, exponent*self.v**(exponent-1)*self.t)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        return Jet(self.v[key], self.t[(slice(None), *key)])

    def sum(self, axis=None, keepdims=False):
        axes = list(range(self.v.shape.rank)) if axis is None else ([axis] if isinstance(axis, int) else axis)
        da = [a+1 if a >= 0 else a for a in axes]
        return Jet(tf.reduce_sum(self.v, axes, keepdims), tf.reduce_sum(self.t, da, keepdims))

    def mean(self, axis, keepdims=False):
        return self.sum(axis, keepdims) / self.v.shape[axis]

    def transpose(self):
        return Jet(tf.linalg.matrix_transpose(self.v), tf.linalg.matrix_transpose(self.t))

    def broadcast(self, shape):
        return Jet(tf.broadcast_to(self.v, shape), tf.broadcast_to(self._aligned(len(shape)), [self.t.shape[0], *shape]))

    def reshape(self, shape):
        return Jet(tf.reshape(self.v, shape), tf.reshape(self.t, [self.t.shape[0], *shape]))

    def gather(self, indices, axis=0):
        return Jet(tf.gather(self.v, indices, axis=axis), tf.gather(self.t, indices, axis=axis+1 if axis>=0 else axis))


def stack(values, axis=0):
    return Jet(tf.stack([x.v for x in values], axis), tf.stack([x.t for x in values], axis+1 if axis>=0 else axis))


def concat(values, axis=0):
    return Jet(tf.concat([x.v for x in values], axis), tf.concat([x.t for x in values], axis+1 if axis>=0 else axis))


def mm(a, b):
    value = tf.linalg.matmul(a.v, b.v)
    rank = value.shape.rank
    return Jet(value, tf.linalg.matmul(a._aligned(rank), b.v) + tf.linalg.matmul(a.v, b._aligned(rank)))


def mv(a, b):
    return mm(a, b[..., None])[..., 0]


def sym(a):
    return .5*(a+a.transpose())


def solve(a, b, triangular=False):
    """Solve the primal, then all partial right-hand sides in a packed solve."""
    batch = tf.broadcast_static_shape(a.v.shape[:-2], b.v.shape[:-2]).as_list()
    a = a.broadcast([*batch, *a.v.shape[-2:]])
    b = b.broadcast([*batch, *b.v.shape[-2:]])
    solver = tf.linalg.triangular_solve if triangular else tf.linalg.solve
    value = solver(a.v, b.v)
    rhs = b.t-tf.linalg.matmul(a.t, value)
    # [P, batch..., M, K] -> [batch..., M, P*K].
    rank = rhs.shape.rank
    packed = tf.reshape(tf.transpose(rhs, [*range(1,rank-1), 0, rank-1]), [*batch, rhs.shape[-2], -1])
    derivative = solver(a.v, packed)
    unpacked = tf.reshape(derivative, [*batch, rhs.shape[-2], rhs.shape[0], rhs.shape[-1]])
    return Jet(value, tf.transpose(unpacked, [rank-2, *range(rank-2), rank-1]))


def cholesky_tangent(value, covariance_tangent):
    # Pack all columns into one factor solve via the same explicit solve rule.
    p = covariance_tangent.shape[0]
    factor = Jet(value, tf.zeros([p, *value.shape], value.dtype))
    perturbation = Jet(tf.zeros_like(value), covariance_tangent)
    first = solve(factor, perturbation, triangular=True)
    white = solve(factor, first.transpose(), triangular=True).transpose().t
    lower = tf.linalg.band_part(white,-1,0)-.5*tf.linalg.diag(tf.linalg.diag_part(white))
    return tf.linalg.matmul(value, lower)


def logsumexp(x, axis, keepdims=False):
    return Jet(tf.reduce_logsumexp(x.v, axis, keepdims),
               tf.reduce_sum(tf.nn.softmax(x.v,axis=axis)*x.t, axis=axis+1 if axis>=0 else axis, keepdims=keepdims))


def exp(x):
    value = tf.exp(x.v)
    return Jet(value, value*x.t)


def log(x):
    return Jet(tf.math.log(x.v), x.t/x.v)


def sqrt(x):
    value = tf.sqrt(x.v)
    return Jet(value, .5*x.t/value)
